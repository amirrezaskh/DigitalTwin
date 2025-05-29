'use strict';

const { TextDecoder } = require("util");

class ModelApp {
    constructor() {
        this.utf8decoder = new TextDecoder();
    }

    async initModels(contract, numNodes) {
        try {
            await (await contract).submitTransaction("InitModels", numNodes);
            return "Models ledger is successfully created.";
        } catch (error) {
            console.log(error);
            return error;
        }
    }

    async readModel(contract, id) {
        try {
            const modelBytes = await (await contract).evaluateTransaction("ReadModel", id);
            const modelString = this.utf8decoder.decode(modelBytes);
            return JSON.parse(modelString);
        } catch (error) {
            console.log(error);
            return error;
        }
    }

    async createModel(contract, id, path, tries=5) {
        while (tries != 0) {
            try {
                const resBytes = await (await contract).submitTransaction("CreateModel", id, path);
                const resString = this.utf8decoder.decode(resBytes);
                return JSON.parse(resString);
            } catch (error) {
                console.log(error)
                tries = tries - 1;
                await new Promise(resolve => setTimeout(resolve, 100));
            }
        }
        console.log("MVCC READ Conflict");
        return false;
    }

    async getAllModels(contract) {
        try {
            const modelsBytes = await (await contract).submitTransaction("GetAllModels");
            const modelsString = this.utf8decoder.decode(modelsBytes);
            return JSON.parse(modelsString);
        } catch (error) {
            console.log(error);
            return error;
        }
    }
}

module.exports = {
    ModelApp
};