# Test Network - Blockchain Infrastructure for Digital Twin Federated Learning

This directory contains the Hyperledger Fabric blockchain network infrastructure that underlies the federated learning Digital Twin system. It provides a permissioned blockchain network for secure model parameter storage and consensus-based aggregation.

## Overview

The test-network implements a development-ready Hyperledger Fabric blockchain network that serves as the distributed ledger foundation for the federated learning system. It manages trust, consensus, and immutable storage of model parameters while ensuring privacy and security.

## Network Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Orderer Node  │    │   Peer0.Org1    │    │   Peer0.Org2    │
│   (RAFT)        │◄──►│   (Endorser)    │◄──►│   (Endorser)    │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   mychannel     │
                    │   (Ledger)      │
                    └─────────────────┘
```

## Key Files

### `network.sh`
**Primary network management script** for controlling the entire blockchain infrastructure:

#### **Core Functions:**
```bash
# Network lifecycle management
./network.sh up                    # Start the network
./network.sh down                  # Stop and cleanup network
./network.sh restart               # Restart the network

# Channel management
./network.sh createChannel         # Create mychannel
./network.sh createChannel -c custom # Create custom channel

# Chaincode deployment
./network.sh deployCC -ccn models  # Deploy model-transfer chaincode
./network.sh deployCC -ccn basic   # Deploy asset-transfer chaincode
```

#### **Advanced Options:**
```bash
# Certificate Authority integration
./network.sh up -ca               # Start with CA for certificate management

# Database selection
./network.sh up -s couchdb        # Use CouchDB for rich queries

# Multi-organization setup
./network.sh up -o org3           # Add third organization

# Custom configuration
./network.sh up -c mychannel -ca -s couchdb
```

### `start.sh`
**Automated startup script** specifically configured for the Digital Twin federated learning system:

#### **Functionality:**
- Brings up the complete Hyperledger Fabric network
- Creates the default mychannel for ledger operations
- Deploys the model-transfer chaincode automatically
- Configures network for federated learning operations
- Sets up proper permissions and policies

#### **Network Configuration:**
```bash
# Network startup sequence:
1. Generate crypto materials (certificates, keys)
2. Start orderer node (RAFT consensus)
3. Start peer nodes (Org1, Org2)
4. Create and join mychannel
5. Deploy model-transfer chaincode
6. Initialize chaincode with default state
```

### `req.sh`
**Chaincode interaction script** for federated learning operations:

#### **Model Management Operations:**
```bash
# Initialize model training rounds
peer chaincode invoke -o localhost:7050 \
  --ordererTLSHostnameOverride orderer.example.com \
  --tls --cafile crypto-config/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem \
  -C mychannel -n models \
  --peerAddresses localhost:7051 \
  --tlsRootCertFiles crypto-config/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt \
  -c '{"function":"InitializeRound","Args":["1"]}'

# Submit model parameters from federated nodes
peer chaincode invoke [...] \
  -c '{"function":"SubmitModelParameters","Args":["node0","<parameters>","1"]}'

# Trigger model aggregation
peer chaincode invoke [...] \
  -c '{"function":"AggregateModelParameters","Args":["1"]}'
```

## Running the Standard Test Network

You can use the `./network.sh` script to stand up a simple Fabric test network. The test network has two peer organizations with one peer each and a single node raft ordering service. You can also use the `./network.sh` script to create channels and deploy chaincode. For more information, see [Using the Fabric test network](https://hyperledger-fabric.readthedocs.io/en/latest/test_network.html). The test network is being introduced in Fabric v2.0 as the long term replacement for the `first-network` sample.

If you are planning to run the test network with consensus type BFT then please pass `-bft` flag as input to the `network.sh` script when creating the channel. Note that currently this sample does not yet support the use of consensus type BFT and CA together.
That is to create a network use:
```bash
./network.sh up -bft
```

To create a channel use:

```bash
./network.sh createChannel -bft
```

To restart a running network use:

```bash
./network.sh restart -bft
```

Note that running the createChannel command will start the network, if it is not already running.

Before you can deploy the test network, you need to follow the instructions to [Install the Samples, Binaries and Docker Images](https://hyperledger-fabric.readthedocs.io/en/latest/install.html) in the Hyperledger Fabric documentation.

## Using the Peer commands

The `setOrgEnv.sh` script can be used to set up the environment variables for the organizations, this will help to be able to use the `peer` commands directly.

First, ensure that the peer binaries are on your path, and the Fabric Config path is set assuming that you're in the `test-network` directory.

```bash
 export PATH=$PATH:$(realpath ../bin)
 export FABRIC_CFG_PATH=$(realpath ../config)
```

You can then set up the environment variables for each organization. The `./setOrgEnv.sh` command is designed to be run as follows.

```bash
export $(./setOrgEnv.sh Org2 | xargs)
```

(Note bash v4 is required for the scripts.)

You will now be able to run the `peer` commands in the context of Org2. If a different command prompt, you can run the same command with Org1 instead.
The `setOrgEnv` script outputs a series of `<name>=<value>` strings. These can then be fed into the export command for your current shell.

## Chaincode-as-a-service

To learn more about how to use the improvements to the Chaincode-as-a-service please see this [tutorial](./test-network/../CHAINCODE_AS_A_SERVICE_TUTORIAL.md). It is expected that this will move to augment the tutorial in the [Hyperledger Fabric ReadTheDocs](https://hyperledger-fabric.readthedocs.io/en/release-2.4/cc_service.html)


## Podman

*Note - podman support should be considered experimental but the following has been reported to work with podman 4.1.1 on Mac. If you wish to use podman a LinuxVM is recommended.*

Fabric's `install-fabric.sh` script has been enhanced to support using `podman` to pull down images and tag them rather than docker. The images are the same, just pulled differently. Simply specify the 'podman' argument when running the `install-fabric.sh` script. 

Similarly, the `network.sh` script has been enhanced so that it can use `podman` and `podman-compose` instead of docker. Just set the environment variable `CONTAINER_CLI` to `podman` before running the `network.sh` script:

```bash
CONTAINER_CLI=podman ./network.sh up
````

As there is no Docker-Daemon when using podman, only the `./network.sh deployCCAAS` command will work. Following the Chaincode-as-a-service Tutorial above should work. 


