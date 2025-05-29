'use strict';

const stringify = require("json-stringify-deterministic");
const sortKeysRecursive = require("sort-keys-recursive");
const { Contract } = require("fabric-contract-api");

class ModelTransfer extends Contract {
    async InitModels(ctx, numNodes) {
        const modelsInfo = {
            id: "modelsInfo",
            numNodes: parseInt(numNodes),
            remaining: parseInt(numNodes)
        }

        await ctx.stub.putState(modelsInfo.id, Buffer.from(stringify(sortKeysRecursive(modelsInfo))));
    }

    async UpdateModelsInfo(ctx) {
        const modelsInfoBinary = await ctx.stub.getState("modelsInfo");
        const modelsInfoString = modelsInfoBinary.toString();
        const modelsInfo = JSON.parse(modelsInfoString);
        modelsInfo.remaining = modelsInfo.remaining - 1;
        if (modelsInfo.remaining === modelsInfo.numNodes) {
            modelsInfo.remaining = modelsInfo.numNodes;
        }
        return modelsInfo.remaining === modelsInfo.numNodes;
    }

    async ReadModel(ctx, id) {
        const modelBinary = await ctx.stub.getState(id);
        if (!modelBinary || modelBinary.length === 0) {
            throw Error(`No model exists with id ${id}`);
        }
        return modelBinary.toString();
    }

    async CreateModel(ctx, id, path) {
        const model = {
            id: id,
            path: path
        }
        await ctx.stub.putState(id, Buffer.from(stringify(sortKeysRecursive(model))));
        const res = await this.UpdateModelsInfo(ctx);
        return JSON.stringify(res);
    }

    async GetAllModels(ctx) {
        const allResults = [];
        const iterator = await ctx.stub.getStateByRange('', '');
        let result = await iterator.next();
        while (!result.done) {
            const strValue = Buffer.from(result.value.value.toString()).toString('utf8');
            let record;
            try {
                record = JSON.parse(strValue);
            } catch (err) {
                console.log(err);
                record = strValue;
            }
            if (record.id.startsWith("model_")) {
                allResults.push(record);
            }
            result = await iterator.next();
        }
        return JSON.stringify(allResults);
    }
}

module.exports = ModelTransfer;