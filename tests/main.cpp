#ifndef __cplusplus
# include <stdatomic.h>
#else
# include <atomic>
# define _Atomic(X) std::atomic< X >
#endif

#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>
