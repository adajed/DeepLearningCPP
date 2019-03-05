# Deep Learning Library in C++

Toy deep learning library written in C++ with support for CUDA.

## Getting Started

To get started clone this repository.

```
git clone https://github.com/adajed/DeepLearningCPP.git
```

### Prerequisites

- CUDA 9.0
- g++ 7.3.0

Additional prerequisites for development:
- clang-format 6.0.0
- clang-tidy 6.0.0

### Installing

To compile just run make from the main directory:
```
make -j4
```
This will compile everything: library, tests and samples.

## Running the tests

```
./build/test
```

Or you can ran a debug version:
```
./build/test_debug
```

## Samples

Source code for samples could be found in **samples** directory.
To run them you have to add compiled library to paths:
```
export LD_LIBRARY_PATH=/path/to/repository/build:$LD_LIBRARY_PATH
```

Example for running mnist sample:
```
./build/sample_mnist
```

## Authors

* **Adam Jędrych** - [adajed](https://github.com/adajed)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
