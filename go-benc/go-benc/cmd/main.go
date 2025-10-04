package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"fisco/store" // import store

	"github.com/FISCO-BCOS/go-sdk/client"
	"github.com/FISCO-BCOS/go-sdk/conf"
	"github.com/ethereum/go-ethereum/common"
)

var (
	constract = ""
	clientx   *client.Client
)

func init() {
	configs, err := conf.ParseConfigFile("config.toml")
	if err != nil {
		log.Fatalf("ParseConfigFile failed, err: %v", err)
	}
	clientx, err = client.Dial(&configs[0])
	if err != nil {
		log.Fatal(err)
	}
	// deploy Contract
	address, _, err := deployContract(clientx)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("contract address: ", address.Hex())
	constract = address.Hex()
}

// 部署合约
func deployContract(client *client.Client) (address common.Address, instance *store.Store, err error) {
	// 部署
	input := "Store deployment 1.0"
	address, tx, instance, err := store.DeployStore(client.GetTransactOpts(), client, input)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("contract address: ", address.Hex()) // the address should be saved, will use in next example
	fmt.Println("transaction hash: ", tx.Hash().Hex())
	return
}

func main() {
	// load the contract
	contractAddress := common.HexToAddress(constract)
	instance, err := store.NewStore(contractAddress, clientx)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("================================")
	storeSession := &store.StoreSession{
		Contract:     instance,
		CallOpts:     *clientx.GetCallOpts(),
		TransactOpts: *clientx.GetTransactOpts(),
	}

	version, err := storeSession.Version()
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("version :", version) // "Store deployment 1.0"

	// 每隔2分钟执行一次 storeSession.SetItem(key, value)
	timmer := time.NewTicker(1 * time.Minute)
	for {
		// 随机数+ 时间戳
		num := rand.Intn(100)
		keyStr := fmt.Sprintf("%d + %v", time.Now().UnixNano(), num)
		select {
		case <-timmer.C:
			fmt.Println("================================")
			key := [32]byte{}
			value := [32]byte{}
			copy(key[:], []byte(keyStr))
			copy(value[:], []byte(keyStr))
			_, _, err := storeSession.SetItem(key, value)
			if err != nil {
				log.Fatal(err)
			}

			// read the result
			result, err := storeSession.Items(key)
			fmt.Println("get item: " + string(result[:]))
			if err != nil {
				log.Fatal(err)
			}
		}
		fmt.Println("================================")
	}
}
