<div align="center">

# 🛡️ Decentralized Fake News Detection Registry

### Immutable News Verification on the Ethereum Blockchain

<p align="center">
  <img src="https://img.shields.io/badge/Blockchain-Ethereum%20Sepolia-3C3C3D?style=for-the-badge&logo=ethereum" alt="Ethereum">
  <img src="https://img.shields.io/badge/Backend-Flask-000000?style=for-the-badge&logo=flask" alt="Flask">
  <img src="https://img.shields.io/badge/Language-Python%203.8%2B-3776AB?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Smart%20Contract-Solidity-363636?style=for-the-badge&logo=solidity" alt="Solidity">
  <img src="https://img.shields.io/badge/License-Proprietary-red?style=for-the-badge" alt="License">
</p>

**Author:** MD. Faisal Ahammad  
**Copyright:** © 2025 MD. Faisal Ahammad. All Rights Reserved.

[Overview](#-overview) • [Features](#-features) • [Tech Stack](#-technology-stack) • [Installation](#-installation--setup) • [API Docs](#-api-documentation)

</div>

---

## 📖 Overview

**Decentralized Fake News Detection** is a blockchain-based system designed to immutably store and verify the credibility of news sources. Unlike traditional centralized databases that can be tampered with, this system utilizes the **Ethereum blockchain** to create a permanent, censorship-resistant registry of news publishers and their source URLs.

The system features a **Flask (Python) Backend** that acts as a secure bridge between web clients and the **Ethereum Blockchain (Sepolia Testnet)**, utilizing **Web3.py** for transaction management and optimized gas estimation.

## 🚀 Features

- **Immutable Registry:** News source data (URL & Publisher) is stored permanently on the blockchain.
- **Smart Contract Logic:**
  - **Duplicate Prevention:** Uses on-chain `keccak256` hashing to ensure a URL cannot be registered twice.
  - **Owner-Only Access:** Only the system administrator can register new sources.
- **Dual Lookup System:**
  - **Fetch by ID:** Retrieve source details using the sequential numeric ID.
  - **Fetch by URL:** Retrieve details using the URL string itself (O(1) lookup cost).
- **Cost-Efficient:** Implements **EIP-1559** dynamic fee estimation.
- **Robust Error Handling:** Manages contract logic errors gracefully.

## 🛠️ Technology Stack

| Component | Technology |
| :--- | :--- |
| **Blockchain** | Solidity (Ethereum / Sepolia Testnet) |
| **Backend** | Python 3.x, Flask |
| **Interface** | Web3.py |
| **Config** | Dotenv |

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.8+ installed.
- An Ethereum Wallet (e.g., MetaMask) with **Sepolia Testnet ETH**.
- An RPC Provider URL (e.g., Infura, Alchemy).

### 2. Clone & Install
```bash
# Clone the repository
git clone [https://github.com/heyahammad/ethereumFake.git](https://github.com/heyahammad/ethereumFake.git)
cd ethereumFake

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt