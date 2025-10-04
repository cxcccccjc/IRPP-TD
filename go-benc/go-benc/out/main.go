package main

import (
	"fmt"
	"log"
	"math/big"
	"sync"
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
	address, tx, instance, err := store.DeployStore(client.GetTransactOpts(), client)
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

	total := 1000

	var time_chain chan float64 = make(chan float64, total)
	// 毫秒时间
	time_old := time.Now().UnixNano()
	var g sync.WaitGroup
	for i := 0; i < total; i++ {
		g.Add(1)
		go func() {
			defer g.Done()
			time_old := time.Now().UnixNano()
			key := [32]byte{}
			copy(key[:], []byte("fdsafdsa"))
			var num big.Int = *big.NewInt(0)
			_, err := storeSession.Verify([5]*big.Int{&num, &num, &num, &num, &num}, key)
			if err != nil {
				log.Fatal(err)
			}
			// 交易hash

			time_new := time.Now().UnixNano()
			cumulative := float64(time_new-time_old) / 1e6
			fmt.Printf("交易执行成功:  block: 1  tx: 1  time: %v \n", cumulative)
			time_chain <- cumulative
		}()
	}
	g.Wait()
	time_new := time.Now().UnixNano()
	cumulative := float64(time_new-time_old) / 1e6

	fmt.Printf(
		"交易总数tx : %d   交易总耗时duration: %v ms   tps: %s   \n",
		total,
		cumulative,
		fmt.Sprintf("%.2f", float64(total)/cumulative*1000),
	)
	// 计算平均时间
	var sum float64
	for i := 0; i < total; i++ {
		sum += <-time_chain
	}
	fmt.Printf("合约平均执行时间: %v ms\n", sum/float64(total))
}
