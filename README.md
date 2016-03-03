# lars::ndarray
A fast c++11 n-dimensional array template

## Notes
- lars::ndarray relies heavily on compiler optimization, so there will be a huge speed difference if run with optimization flags enabled

## Run the example

        g++ --std=c++11 -O3 -DNDEBUG timeit.cpp && ./a.out
        
or using cmake:

        cmake -DCMAKE_BUILD_TYPE=Release . && make && ./a.out



## Documentation
Stay tuned for more
