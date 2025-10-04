# 生成abi文件
./solc-0.4.25 --bin --abi -o ./schnorr /Users/rock/myself/gitdir/ether-schnorr-verification/contracts/SchnorrOptimized.sol

# 生成go
./abigen --bin ./schnorr/SchnorrOptimized.bin --abi ./schnorr/SchnorrOptimized.abi --pkg store --type Store --out ./schnorr/SchnorrOptimized.go
