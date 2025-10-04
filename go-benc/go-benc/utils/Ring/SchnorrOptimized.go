// Code generated - DO NOT EDIT.
// This file is a generated binding and any manual changes will be lost.

package store

import (
	"math/big"
	"strings"

	"github.com/FISCO-BCOS/go-sdk/abi"
	"github.com/FISCO-BCOS/go-sdk/abi/bind"
	"github.com/FISCO-BCOS/go-sdk/core/types"
	ethereum "github.com/ethereum/go-ethereum"
	"github.com/ethereum/go-ethereum/common"
)

// Reference imports to suppress errors if they are not otherwise used.
var (
	_ = big.NewInt
	_ = strings.NewReader
	_ = ethereum.NotFound
	_ = abi.U256
	_ = bind.Bind
	_ = common.Big1
	_ = types.BloomLookup
)

// StoreABI is the input ABI used to generate the binding from.
const StoreABI = "[{\"constant\":true,\"inputs\":[{\"name\":\"a\",\"type\":\"uint256\"},{\"name\":\"b\",\"type\":\"uint256\"},{\"name\":\"c\",\"type\":\"uint256\"},{\"name\":\"d\",\"type\":\"uint256\"},{\"name\":\"m\",\"type\":\"bytes32\"}],\"name\":\"h\",\"outputs\":[{\"name\":\"\",\"type\":\"uint256\"}],\"payable\":false,\"stateMutability\":\"view\",\"type\":\"function\"},{\"constant\":true,\"inputs\":[{\"name\":\"_musig\",\"type\":\"uint256[5]\"},{\"name\":\"_message\",\"type\":\"bytes32\"}],\"name\":\"verify\",\"outputs\":[{\"name\":\"\",\"type\":\"bool\"}],\"payable\":false,\"stateMutability\":\"view\",\"type\":\"function\"}]"

// StoreBin is the compiled bytecode used for deploying new contracts.
var StoreBin = "0x608060405234801561001057600080fd5b50610691806100206000396000f30060806040526004361061004c576000357c0100000000000000000000000000000000000000000000000000000000900463ffffffff16806351fc3ba91461005157806397d88f631461008e575b600080fd5b34801561005d57600080fd5b5061007860048036036100739190810190610513565b6100cb565b60405161008591906105c3565b60405180910390f35b34801561009a57600080fd5b506100b560048036036100b091908101906104d7565b61011a565b6040516100c291906105a8565b60405180910390f35b6000858585858560405180868152602001858152602001848152602001838152602001826000191660001916815260200195505050505050604051809103902060019004905095945050505050565b6000806101256103d9565b61012d6103d9565b61018686600160058110151561013f57fe5b602002015187600260058110151561015357fe5b602002015188600360058110151561016757fe5b602002015189600460058110151561017b57fe5b6020020151896100cb565b92506101ac610193610260565b8760006005811015156101a257fe5b6020020151610284565b915061023560408051908101604052808860016005811015156101cb57fe5b602002015181526020018860026005811015156101e457fe5b602002015181525061023060408051908101604052808a600360058110151561020957fe5b602002015181526020018a600460058110151561022257fe5b602002015181525086610284565b61031e565b905080600001518260000151148015610255575080602001518260200151145b935050505092915050565b6102686103d9565b6040805190810160405280600181526020016002815250905090565b61028c6103d9565b6102946103f3565b600084600001518260006003811015156102aa57fe5b60200201818152505084602001518260016003811015156102c757fe5b602002018181525050838260026003811015156102e057fe5b60200201818152505060608360808460076107d05a03fa9050806000811461030757610309565bfe5b5080151561031657600080fd5b505092915050565b6103266103d9565b61032e610416565b6000846000015182600060048110151561034457fe5b602002018181525050846020015182600160048110151561036157fe5b602002018181525050836000015182600260048110151561037e57fe5b602002018181525050836020015182600360048110151561039b57fe5b60200201818152505060608360c08460066107d05a03fa905080600081146103c2576103c4565bfe5b508015156103d157600080fd5b505092915050565b604080519081016040528060008152602001600081525090565b606060405190810160405280600390602082028038833980820191505090505090565b608060405190810160405280600490602082028038833980820191505090505090565b600082601f830112151561044c57600080fd5b600561045f61045a8261060b565b6105de565b9150818385602084028201111561047557600080fd5b60005b838110156104a5578161048b88826104c3565b845260208401935060208301925050600181019050610478565b5050505092915050565b60006104bb8235610643565b905092915050565b60006104cf823561064d565b905092915050565b60008060c083850312156104ea57600080fd5b60006104f885828601610439565b92505060a0610509858286016104af565b9150509250929050565b600080600080600060a0868803121561052b57600080fd5b6000610539888289016104c3565b955050602061054a888289016104c3565b945050604061055b888289016104c3565b935050606061056c888289016104c3565b925050608061057d888289016104af565b9150509295509295909350565b6105938161062d565b82525050565b6105a281610639565b82525050565b60006020820190506105bd600083018461058a565b92915050565b60006020820190506105d86000830184610599565b92915050565b6000604051905081810181811067ffffffffffffffff8211171561060157600080fd5b8060405250919050565b600067ffffffffffffffff82111561062257600080fd5b602082029050919050565b60008115159050919050565b6000819050919050565b6000819050919050565b60008190509190505600a265627a7a723058208b1b391d38c3dc33efad37e1cf222533b5fd5d19d06a6fbb6fa661e96e03c3566c6578706572696d656e74616cf50037"

// DeployStore deploys a new contract, binding an instance of Store to it.
func DeployStore(auth *bind.TransactOpts, backend bind.ContractBackend) (common.Address, *types.Transaction, *Store, error) {
	parsed, err := abi.JSON(strings.NewReader(StoreABI))
	if err != nil {
		return common.Address{}, nil, nil, err
	}

	address, tx, contract, err := bind.DeployContract(auth, parsed, common.FromHex(StoreBin), backend)
	if err != nil {
		return common.Address{}, nil, nil, err
	}
	return address, tx, &Store{StoreCaller: StoreCaller{contract: contract}, StoreTransactor: StoreTransactor{contract: contract}, StoreFilterer: StoreFilterer{contract: contract}}, nil
}

func AsyncDeployStore(auth *bind.TransactOpts, handler func(*types.Receipt, error), backend bind.ContractBackend) (*types.Transaction, error) {
	parsed, err := abi.JSON(strings.NewReader(StoreABI))
	if err != nil {
		return nil, err
	}

	tx, err := bind.AsyncDeployContract(auth, handler, parsed, common.FromHex(StoreBin), backend)
	if err != nil {
		return nil, err
	}
	return tx, nil
}

// Store is an auto generated Go binding around a Solidity contract.
type Store struct {
	StoreCaller     // Read-only binding to the contract
	StoreTransactor // Write-only binding to the contract
	StoreFilterer   // Log filterer for contract events
}

// StoreCaller is an auto generated read-only Go binding around a Solidity contract.
type StoreCaller struct {
	contract *bind.BoundContract // Generic contract wrapper for the low level calls
}

// StoreTransactor is an auto generated write-only Go binding around a Solidity contract.
type StoreTransactor struct {
	contract *bind.BoundContract // Generic contract wrapper for the low level calls
}

// StoreFilterer is an auto generated log filtering Go binding around a Solidity contract events.
type StoreFilterer struct {
	contract *bind.BoundContract // Generic contract wrapper for the low level calls
}

// StoreSession is an auto generated Go binding around a Solidity contract,
// with pre-set call and transact options.
type StoreSession struct {
	Contract     *Store            // Generic contract binding to set the session for
	CallOpts     bind.CallOpts     // Call options to use throughout this session
	TransactOpts bind.TransactOpts // Transaction auth options to use throughout this session
}

// StoreCallerSession is an auto generated read-only Go binding around a Solidity contract,
// with pre-set call options.
type StoreCallerSession struct {
	Contract *StoreCaller  // Generic contract caller binding to set the session for
	CallOpts bind.CallOpts // Call options to use throughout this session
}

// StoreTransactorSession is an auto generated write-only Go binding around a Solidity contract,
// with pre-set transact options.
type StoreTransactorSession struct {
	Contract     *StoreTransactor  // Generic contract transactor binding to set the session for
	TransactOpts bind.TransactOpts // Transaction auth options to use throughout this session
}

// StoreRaw is an auto generated low-level Go binding around a Solidity contract.
type StoreRaw struct {
	Contract *Store // Generic contract binding to access the raw methods on
}

// StoreCallerRaw is an auto generated low-level read-only Go binding around a Solidity contract.
type StoreCallerRaw struct {
	Contract *StoreCaller // Generic read-only contract binding to access the raw methods on
}

// StoreTransactorRaw is an auto generated low-level write-only Go binding around a Solidity contract.
type StoreTransactorRaw struct {
	Contract *StoreTransactor // Generic write-only contract binding to access the raw methods on
}

// NewStore creates a new instance of Store, bound to a specific deployed contract.
func NewStore(address common.Address, backend bind.ContractBackend) (*Store, error) {
	contract, err := bindStore(address, backend, backend, backend)
	if err != nil {
		return nil, err
	}
	return &Store{StoreCaller: StoreCaller{contract: contract}, StoreTransactor: StoreTransactor{contract: contract}, StoreFilterer: StoreFilterer{contract: contract}}, nil
}

// NewStoreCaller creates a new read-only instance of Store, bound to a specific deployed contract.
func NewStoreCaller(address common.Address, caller bind.ContractCaller) (*StoreCaller, error) {
	contract, err := bindStore(address, caller, nil, nil)
	if err != nil {
		return nil, err
	}
	return &StoreCaller{contract: contract}, nil
}

// NewStoreTransactor creates a new write-only instance of Store, bound to a specific deployed contract.
func NewStoreTransactor(address common.Address, transactor bind.ContractTransactor) (*StoreTransactor, error) {
	contract, err := bindStore(address, nil, transactor, nil)
	if err != nil {
		return nil, err
	}
	return &StoreTransactor{contract: contract}, nil
}

// NewStoreFilterer creates a new log filterer instance of Store, bound to a specific deployed contract.
func NewStoreFilterer(address common.Address, filterer bind.ContractFilterer) (*StoreFilterer, error) {
	contract, err := bindStore(address, nil, nil, filterer)
	if err != nil {
		return nil, err
	}
	return &StoreFilterer{contract: contract}, nil
}

// bindStore binds a generic wrapper to an already deployed contract.
func bindStore(address common.Address, caller bind.ContractCaller, transactor bind.ContractTransactor, filterer bind.ContractFilterer) (*bind.BoundContract, error) {
	parsed, err := abi.JSON(strings.NewReader(StoreABI))
	if err != nil {
		return nil, err
	}
	return bind.NewBoundContract(address, parsed, caller, transactor, filterer), nil
}

// Call invokes the (constant) contract method with params as input values and
// sets the output to result. The result type might be a single field for simple
// returns, a slice of interfaces for anonymous returns and a struct for named
// returns.
func (_Store *StoreRaw) Call(opts *bind.CallOpts, result interface{}, method string, params ...interface{}) error {
	return _Store.Contract.StoreCaller.contract.Call(opts, result, method, params...)
}

// Transfer initiates a plain transaction to move funds to the contract, calling
// its default method if one is available.
func (_Store *StoreRaw) Transfer(opts *bind.TransactOpts) (*types.Transaction, *types.Receipt, error) {
	return _Store.Contract.StoreTransactor.contract.Transfer(opts)
}

// Transact invokes the (paid) contract method with params as input values.
func (_Store *StoreRaw) TransactWithResult(opts *bind.TransactOpts, result interface{}, method string, params ...interface{}) (*types.Transaction, *types.Receipt, error) {
	return _Store.Contract.StoreTransactor.contract.TransactWithResult(opts, result, method, params...)
}

// Call invokes the (constant) contract method with params as input values and
// sets the output to result. The result type might be a single field for simple
// returns, a slice of interfaces for anonymous returns and a struct for named
// returns.
func (_Store *StoreCallerRaw) Call(opts *bind.CallOpts, result interface{}, method string, params ...interface{}) error {
	return _Store.Contract.contract.Call(opts, result, method, params...)
}

// Transfer initiates a plain transaction to move funds to the contract, calling
// its default method if one is available.
func (_Store *StoreTransactorRaw) Transfer(opts *bind.TransactOpts) (*types.Transaction, *types.Receipt, error) {
	return _Store.Contract.contract.Transfer(opts)
}

// Transact invokes the (paid) contract method with params as input values.
func (_Store *StoreTransactorRaw) TransactWithResult(opts *bind.TransactOpts, result interface{}, method string, params ...interface{}) (*types.Transaction, *types.Receipt, error) {
	return _Store.Contract.contract.TransactWithResult(opts, result, method, params...)
}

// H is a free data retrieval call binding the contract method 0x51fc3ba9.
//
// Solidity: function h(uint256 a, uint256 b, uint256 c, uint256 d, bytes32 m) constant returns(uint256)
func (_Store *StoreCaller) H(opts *bind.CallOpts, a *big.Int, b *big.Int, c *big.Int, d *big.Int, m [32]byte) (*big.Int, error) {
	var (
		ret0 = new(*big.Int)
	)
	out := ret0
	err := _Store.contract.Call(opts, out, "h", a, b, c, d, m)
	return *ret0, err
}

// H is a free data retrieval call binding the contract method 0x51fc3ba9.
//
// Solidity: function h(uint256 a, uint256 b, uint256 c, uint256 d, bytes32 m) constant returns(uint256)
func (_Store *StoreSession) H(a *big.Int, b *big.Int, c *big.Int, d *big.Int, m [32]byte) (*big.Int, error) {
	return _Store.Contract.H(&_Store.CallOpts, a, b, c, d, m)
}

// H is a free data retrieval call binding the contract method 0x51fc3ba9.
//
// Solidity: function h(uint256 a, uint256 b, uint256 c, uint256 d, bytes32 m) constant returns(uint256)
func (_Store *StoreCallerSession) H(a *big.Int, b *big.Int, c *big.Int, d *big.Int, m [32]byte) (*big.Int, error) {
	return _Store.Contract.H(&_Store.CallOpts, a, b, c, d, m)
}

// Verify is a free data retrieval call binding the contract method 0x97d88f63.
//
// Solidity: function verify(uint256[5] _musig, bytes32 _message) constant returns(bool)
func (_Store *StoreCaller) Verify(opts *bind.CallOpts, _musig [5]*big.Int, _message [32]byte) (bool, error) {
	var (
		ret0 = new(bool)
	)
	out := ret0
	err := _Store.contract.Call(opts, out, "verify", _musig, _message)
	return *ret0, err
}

// Verify is a free data retrieval call binding the contract method 0x97d88f63.
//
// Solidity: function verify(uint256[5] _musig, bytes32 _message) constant returns(bool)
func (_Store *StoreSession) Verify(_musig [5]*big.Int, _message [32]byte) (bool, error) {
	return _Store.Contract.Verify(&_Store.CallOpts, _musig, _message)
}

// Verify is a free data retrieval call binding the contract method 0x97d88f63.
//
// Solidity: function verify(uint256[5] _musig, bytes32 _message) constant returns(bool)
func (_Store *StoreCallerSession) Verify(_musig [5]*big.Int, _message [32]byte) (bool, error) {
	return _Store.Contract.Verify(&_Store.CallOpts, _musig, _message)
}
