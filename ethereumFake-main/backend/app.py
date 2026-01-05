from flask import Flask, request, jsonify
from web3 import Web3
from web3.exceptions import ContractLogicError # Import specific error handling
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# ---------------- Web3 Setup -----------------
try:
    RPC_URL = os.getenv("RPC_URL")
    PRIVATE_KEY = os.getenv("PRIVATE_KEY")
    CONTRACT_ADDRESS = os.getenv("CONTRACT_ADDRESS")
    # We still load this for validation, but we will trust the RPC node's ID for the transaction
    ENV_CHAIN_ID = int(os.getenv("CHAIN_ID"))
except (TypeError, ValueError) as e:
    print("FATAL: Missing or invalid environment variable. Check your .env file.")
    raise e


web3 = Web3(Web3.HTTPProvider(RPC_URL))
# Check connection status
if not web3.is_connected():
    print(f"FATAL: Failed to connect to RPC URL: {RPC_URL}")
    exit()

account = web3.eth.account.from_key(PRIVATE_KEY)

# --- CRITICAL CHAIN ID CHECK ---
connected_chain_id = web3.eth.chain_id
print(f"✅ Connected to RPC Node. Chain ID: {connected_chain_id}")

if connected_chain_id != ENV_CHAIN_ID:
    print(f"⚠️ WARNING: Mismatch! .env says Chain ID {ENV_CHAIN_ID}, but RPC is {connected_chain_id}.")
    print(f"   -> Transactions will use the RPC's Chain ID ({connected_chain_id}) to prevent errors.")
else:
    print(f"✅ Chain ID matches configuration ({connected_chain_id}).")

# --- BALANCE CHECK (DEBUGGING) ---
balance_wei = web3.eth.get_balance(account.address)
balance_eth = web3.from_wei(balance_wei, 'ether')
print(f"   Account: {account.address}")
print(f"   Balance: {balance_eth} ETH") 

if balance_wei == 0:
    print("❌ CRITICAL: This account has 0 ETH. It cannot send transactions.")
    print("   -> Please check if PRIVATE_KEY in .env matches the wallet with funds.")


# Load contract ABI
try:
    with open("abi.json") as f:
        CONTRACT_ABI = json.load(f)
    contract = web3.eth.contract(
        address=Web3.to_checksum_address(CONTRACT_ADDRESS),
        abi=CONTRACT_ABI
    )
except Exception as e:
    print("FATAL: Could not load ABI or initialize contract. Ensure abi.json exists and CONTRACT_ADDRESS is valid.")
    raise e


# ------------------------------------------------------
# Helper function for EIP-1559 Gas Calculation
# Ensures minimum cost by setting Max Fee only as high as necessary.
# ------------------------------------------------------
def calculate_eip1559_fees():
    """Fetches and calculates reliable EIP-1559 gas fee parameters."""
    try:
        # Get Max Priority Fee (the Tip to the validator) in Wei.
        # This is usually 1 Gwei on testnets, but we fetch dynamically for safety.
        max_priority_fee = web3.eth.max_priority_fee

        # Get current Base Fee for the latest block (in Wei). This is burned.
        base_fee = web3.eth.get_block('latest').baseFeePerGas

        # Calculate Max Fee (Base Fee * 2 + Priority Fee).
        # We use Base Fee * 2 as a safety buffer for the next few blocks.
        max_fee = base_fee * 2 + max_priority_fee
        
        # Ensure max_fee is greater than or equal to max_priority_fee (standard check)
        if max_fee < max_priority_fee:
            max_fee = max_priority_fee * 2
        
        return {
            "maxFeePerGas": max_fee,
            "maxPriorityFeePerGas": max_priority_fee
        }

    except Exception as e:
        print(f"Error calculating EIP-1559 fees, falling back to 2 Gwei base: {e}")
        # Fallback in case RPC fails to return EIP-1559 data
        fallback_gwei = web3.to_wei(2, 'gwei')
        return {
            "maxFeePerGas": fallback_gwei,
            "maxPriorityFeePerGas": fallback_gwei
        }


# ------------------------------------------------------
# POST /register - Registers a new news source
# ------------------------------------------------------
@app.route("/register", methods=["POST"])
def register_source():
    data = request.json
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    try:
        url = data["url"]
        publisher = data["publisher"]
    except Exception:
        return jsonify({"error": "Missing required fields: 'url' and 'publisher'"}), 400

    try:
        # 1. Estimate Gas (The Work)
        # This gives us the gas limit needed for the execution.
        gas_est = contract.functions.registerSource(
            url, publisher
        ).estimate_gas({"from": account.address})
        
        print(f"ℹ️  Gas Estimate: {gas_est}")

        # 2. Calculate EIP-1559 Fees (The Price)
        fees = calculate_eip1559_fees()
        print(f"ℹ️  Fees: {fees}")
        
        # 3. Build Transaction
        # CRITICAL FIX: We use 'web3.eth.chain_id' directly here to ensure the 
        # transaction matches the node we are actually connected to.
        tx = contract.functions.registerSource(
            url, publisher
        ).build_transaction({
            "from": account.address,
            "nonce": web3.eth.get_transaction_count(account.address),
            "gas": gas_est + 20000, # Add a buffer to the estimate for safety
            "chainId": web3.eth.chain_id, # <--- DYNAMIC FIX
            "maxFeePerGas": fees["maxFeePerGas"],
            "maxPriorityFeePerGas": fees["maxPriorityFeePerGas"]
        })

        # 4. Sign & Send
        signed = web3.eth.account.sign_transaction(tx, PRIVATE_KEY)
        tx_hash = web3.eth.send_raw_transaction(signed.raw_transaction)

        # OPTIONAL: wait & confirm
        receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
        
        # Extract the ID generated by the contract (using the nextSourceId value before increment)
        # NOTE: This is a rough estimation. The true ID should be read from the event log.
        source_id = contract.functions.nextSourceId().call() - 1 

    except ContractLogicError as e:
        # This catches "URL already registered" or "Only owner can register" errors
        print(f"❌ Logic Error: {e}")
        return jsonify({"error": "Contract Logic Failed", "details": str(e)}), 400
    except Exception as e:
        print(f"❌ Transaction failed: {e}")
        return jsonify({"error": "Transaction failed", "details": str(e)}), 500

    return jsonify({
        "status": "success",
        "message": "Source registered successfully",
        "source_id": source_id,
        "tx_hash": tx_hash.hex(),
        "gas_used": receipt.gasUsed,
        "effective_gas_price": web3.from_wei(receipt.effectiveGasPrice, 'gwei')
    }), 200


# ------------------------------------------------------
# GET /source?id=... - Retrieves source data
# ------------------------------------------------------
@app.get("/source")
def get_source():
    source_id = request.args.get("id", type=int)
    if source_id is None:
        return jsonify({"error": "Source ID is required"}), 400

    try:
        url, publisher = contract.functions.getSource(source_id).call()
        
        if not url: # Check if the stored URL is empty
            return jsonify({"error": "Source ID not found or data is empty"}), 404

        return jsonify({
            "status": "success",
            "source_id": source_id,
            "url": url,
            "publisher": publisher
        }), 200

    except Exception as e:
        return jsonify({
            "error": "Failed to retrieve source",
            "details": str(e)
        }), 404


# ------------------------------------------------------
# GET /source_by_url?url=... - Retrieves source data by URL
# ------------------------------------------------------
@app.get("/getNews")
def get_source_by_url():
    url_query = request.args.get("url")
    if not url_query:
        return jsonify({"error": "URL parameter is required"}), 400

    try:
        # Call the new contract function
        # This uses the mapping(bytes32 => uint256) logic inside the contract
        retrieved_url, publisher = contract.functions.getSourceByUrl(url_query).call()
        
        return jsonify({
            "status": "success",
            "url": retrieved_url,
            "publisher": publisher
        }), 200

    except Exception as e:
        return jsonify({
            "error": "URL not found in registry",
            "details": str(e)
        }), 404
# ------------------------------------------------------
# GET /getNewsByPublisher?publisher=... 
# Retrieves all URLs for a specific publisher
# ------------------------------------------------------
@app.get("/getNewsByPublisher")
def get_sources_by_publisher():
    publisher_query = request.args.get("publisher")
    
    if not publisher_query:
        return jsonify({"error": "Publisher parameter is required"}), 400

    try:
        # Call the new contract function 'getSourcesByPublisher'
        # The contract returns a list (array) of strings, which web3.py converts to a Python list
        retrieved_urls = contract.functions.getSourcesByPublisher(publisher_query).call()
        
        # If the list is empty, it means no records were found for that publisher
        if not retrieved_urls:
            return jsonify({
                "status": "success",
                "message": f"No sources found for publisher: {publisher_query}",
                "urls": []
            }), 200

        return jsonify({
            "status": "success",
            "publisher": publisher_query,
            "count": len(retrieved_urls),
            "urls": retrieved_urls
        }), 200

    except Exception as e:
        print(f"❌ Error fetching by publisher: {e}")
        return jsonify({
            "error": "Failed to retrieve sources",
            "details": str(e)
        }), 500

# ------------------------------------------------------
# Start Server
# ------------------------------------------------------
if __name__ == "__main__":
    # 1. Get the PORT from Render's environment, default to 5000 for local testing
    port = int(os.environ.get("PORT", 5000))
    
    print(f"Server running → http://0.0.0.0:{port}")
    
    # 2. Host MUST be '0.0.0.0' for Render to see it
    # 3. Disable debug mode for production security
    app.run(host='0.0.0.0', port=port, debug=False)