./network.sh down
./network.sh up
./network.sh createChannel -c main
./network.sh deployCC -ccp ../model-transfer/model-transfer-chaincode -ccn modelsCC -c main -ccl javascript