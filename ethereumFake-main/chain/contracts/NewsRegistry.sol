// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract NewsRegistry {
    address public owner;
    uint256 public nextSourceId;

    struct NewsSource {
        string url;
        string publisher;
    }

    mapping(uint256 => NewsSource) public sources;
    mapping(bytes32 => uint256) public urlHashToSourceId;

    // --- CHANGE 1: New mapping to store a list of IDs for each Publisher ---
    // We map the hash of the publisher name to an array of Source IDs
    mapping(bytes32 => uint256[]) private publisherToIds;

    event SourceRegistered(uint256 indexed id, string url, string publisher);
    event SourceLookedUp(uint256 indexed id, bytes32 indexed urlHash); 

    constructor() {
        owner = msg.sender;
        nextSourceId = 1; 
    }

    function registerSource(
        string calldata _url,
        string calldata _publisher
    ) external {
        require(msg.sender == owner, "Only owner can register.");

        bytes32 urlHash = keccak256(bytes(_url));
        require(urlHashToSourceId[urlHash] == 0, "URL already registered.");

        urlHashToSourceId[urlHash] = nextSourceId;

        // --- CHANGE 2: Update the publisher mapping ---
        // Hash the publisher name to handle string comparison efficiently
        bytes32 pubHash = keccak256(bytes(_publisher));
        publisherToIds[pubHash].push(nextSourceId);

        NewsSource storage newSource = sources[nextSourceId];
        newSource.url = _url;
        newSource.publisher = _publisher;
        
        emit SourceRegistered(nextSourceId, _url, _publisher);

        nextSourceId++;
    }

    function getSource(uint256 _id) external view returns (string memory, string memory) {
        NewsSource storage source = sources[_id];
        require(bytes(source.url).length > 0, "Source ID not found.");
        return (source.url, source.publisher);
    }

    function getSourceByUrl(string calldata _url) external view returns (string memory, string memory) {
        bytes32 urlHash = keccak256(bytes(_url));
        uint256 sourceId = urlHashToSourceId[urlHash];
        require(sourceId != 0, "URL not found in registry.");
        NewsSource storage source = sources[sourceId];
        return (source.url, source.publisher);
    }

    // --- CHANGE 3: New Getter Function for Publisher ---
    /**
     * @notice Returns all URLs registered by a specific publisher.
     * @param _publisher The exact name of the publisher (Case Sensitive).
     */
    function getSourcesByPublisher(string calldata _publisher) external view returns (string[] memory) {
        bytes32 pubHash = keccak256(bytes(_publisher));
        
        // Get the list of IDs associated with this publisher
        uint256[] memory ids = publisherToIds[pubHash];
        
        // Create a temporary array to hold the URLs
        string[] memory urls = new string[](ids.length);

        // Loop through IDs and fetch the URL for each
        for (uint256 i = 0; i < ids.length; i++) {
            urls[i] = sources[ids[i]].url;
        }

        return urls;
    }
}