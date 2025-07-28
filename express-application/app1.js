const { ModelApp } = require("../model-transfer/model-transfer-application/modelApp");
const modelApp = new ModelApp();

const fileService = require('node:fs');
const { exec } = require('child_process');

const express = require('express');
const bodyParser = require('body-parser');
const app = express();
const jsonParser = bodyParser.json();
const port = 3000;

const rounds = 2;
let currentRound = 0;
const aggregatorPort = 8080;
const numNodes = 4;

const crypto = require("crypto");
const grpc = require("@grpc/grpc-js");
const {
    connect,
    Contract,
    Identity,
    Signer,
    signers
} = require("@hyperledger/fabric-gateway");
const fs = require("fs/promises");
const path = require("path");

const mspId = "Org1MSP";

const cryptoPath = path.resolve(__dirname, '..', 'test-network', 'organizations', 'peerOrganizations', 'org1.example.com');
const keyDirPath = path.resolve(cryptoPath, 'users', 'User1@org1.example.com', 'msp', 'keystore');
const certPath = path.resolve(cryptoPath, 'users', 'User1@org1.example.com', 'msp', 'signcerts', 'User1@org1.example.com-cert.pem');
const tlsCertPath = path.resolve(cryptoPath, 'peers', 'peer0.org1.example.com', 'tls', 'ca.crt');

const peerEndPoint = "localhost:7051";
const peerHostAlias = "peer0.org1.example.com";

const contractModels = InitConnection("main", "modelsCC");
const axios = require("axios");
const { start } = require("node:repl");

async function newGrpcConnection() {
    const tlsRootCert = await fs.readFile(tlsCertPath);
    const tlsCredentials = grpc.credentials.createSsl(tlsRootCert);
    return new grpc.Client(peerEndPoint, tlsCredentials, {
        'grpc.ssl_target_name_override': peerHostAlias,
        'grpc.max_send_message_length': 100 * 1024 * 1024,
        'grpc.max_receive_message_length': 100 * 1024 * 1024
    });
}

async function newIdentity() {
    const credentials = await fs.readFile(certPath);
    return {
        mspId,
        credentials
    };
}

async function newSigner() {
    const files = await fs.readdir(keyDirPath);
    const keyPath = path.resolve(keyDirPath, files[0]);
    const privateKeyPem = await fs.readFile(keyPath);
    const privateKey = crypto.createPrivateKey(privateKeyPem);
    return signers.newPrivateKeySigner(privateKey);
}

async function InitConnection(channelName, chaincodeName) {
    /*
     * Returns a contract for a given channel and chaincode.
     * */
    const client = await newGrpcConnection();

    const gateway = connect({
        client,
        identity: await newIdentity(),
        signer: await newSigner(),
        // Default timeouts for different gRPC calls
        evaluateOptions: () => {
            return {
                deadline: Date.now() + 500000
            }; // 5 seconds
        },
        endorseOptions: () => {
            return {
                deadline: Date.now() + 1500000
            }; // 15 seconds
        },
        submitOptions: () => {
            return {
                deadline: Date.now() + 500000
            }; // 5 seconds
        },
        commitStatusOptions: () => {
            return {
                deadline: Date.now() + 6000000
            }; // 1 minute
        },
    });

    const network = gateway.getNetwork(channelName);

    return network.getContract(chaincodeName);
}

async function callAggregator() {
	const models = await modelApp.getAllModels(contractModels);
	await axios({
		method: "post",
		url: `http://localhost:${aggregatorPort}/aggregate/`,
		headers: {},
		data: {
			models: models,
		},
	});
    await startRound();
}

async function startRound() {
    currentRound += 1;
    if (currentRound <= rounds) {
        for (let i = 0; i < numNodes; i++) {
            axios({
                method: 'get',
                url: `http://localhost:${8000 + i}/train/`,
                headers: {}
            });
        }
        console.log(`*** Round ${currentRound} STARTED ***`);
        console.log("Training started.");
    } else {
        await axios.get(`http://localhost:${aggregatorPort}/losses/`);
        console.log("All rounds completed.");
        exec('python3 ../stop.py', (error, stdout, stderr) => {
            if (error) {
              console.error(`Error: ${error.message}`);
              return;
            }
            if (stderr) {
              console.error(`stderr: ${stderr}`);
              return;
            }
            console.log(`stdout: ${stdout}`);
          });
        currentRound = 0;
    }
}

app.get('/', (req, res) => {
    res.send("Hello World!.");
});

app.get('/exit', (req, res) => {
    process.exit();
});

// **** MODEL PROPOSE API ****
app.post('/api/models/ledger/', async (req, res) => {
    const message = await modelApp.initModels(contractModels, numNodes.toString());
    res.send(message);
});

app.post('/api/model/', jsonParser, async (req, res) => {
    const respond = await modelApp.createModel(contractModels, req.body.id, req.body.path);
    console.log(respond)
    if (respond === true) {
        console.log("here")
        setTimeout(callAggregator, 1)
    }
    res.send("Model was created successfully.");
});

app.get('/api/model/', jsonParser, async (req, res) => {
    const message = await modelApp.readModel(contractModels, req.body.id);
    res.send(message);
});

app.get('/api/models/', async (req, res) => {
    const message = await modelApp.getAllModels(contractModels);
    res.send(message);
});

app.get('/api/start/', async (req, res) => {
    await startRound();
    res.send("Rounds started.")
})

app.listen(port, () => {
    console.log(`Server is listening on localhost:${port}.\n`);
});