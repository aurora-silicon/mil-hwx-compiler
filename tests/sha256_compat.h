#pragma once
// SHA-256 test assertions use the platform library, never another encoder.
#if defined(__APPLE__)
#import <CommonCrypto/CommonDigest.h>
#else
#include <openssl/sha.h>
#define CC_SHA256_DIGEST_LENGTH SHA256_DIGEST_LENGTH
using CC_LONG = size_t;
#define CC_SHA256(data, length, digest) \
    SHA256((const unsigned char *)(data), (length), (digest))
#endif
