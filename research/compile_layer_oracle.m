// SPDX-License-Identifier: MIT
// Compile-only ANECCompile wrapper. No runtime evaluation or ANE submission.
#import <Foundation/Foundation.h>
#include <dispatch/dispatch.h>
#include <dlfcn.h>
#include <limits.h>

typedef CFDictionaryRef (*CreateInput)(const char *);
typedef void (*Compile)(CFDictionaryRef,CFDictionaryRef,void (^)(int,CFDictionaryRef));

int main(int argc,const char **argv) {
    @autoreleasepool {
        if (argc!=4) {
            fprintf(stderr,"usage: ane-compile-hwx CAPTURE_DIR OUTPUT_DIR h13|h14\n");
            return 64;
        }
        NSString *target=[NSString stringWithUTF8String:argv[3]];
        NSString *architecture=[@{@"h13":@"H13",@"h14":@"H14"} objectForKey:target.lowercaseString];
        if (!architecture) return 65;
        NSString *capture=[NSString stringWithUTF8String:argv[1]];
        NSString *output=[NSString stringWithUTF8String:argv[2]];
        NSFileManager *files=NSFileManager.defaultManager;
        NSError *error=nil;
        if (![files createDirectoryAtPath:output withIntermediateDirectories:YES attributes:nil error:&error]) return 2;
        void *library=dlopen("/System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler",RTLD_NOW|RTLD_LOCAL);
        CreateInput create=library?(CreateInput)dlsym(library,"ANECCreateCompilerInputDictionary"):NULL;
        Compile compile=library?(Compile)dlsym(library,"ANECCompile"):NULL;
        if (!create || !compile) {
            fprintf(stderr,"Apple compiler entry points unavailable\n");
            return 3;
        }
        CFDictionaryRef base=create([capture stringByAppendingPathComponent:@"model.mil"].fileSystemRepresentation);
        if (!base) return 4;
        CFMutableDictionaryRef input=CFDictionaryCreateMutableCopy(kCFAllocatorDefault,0,base);
        CFDictionarySetValue(input,CFSTR("ModelPath"),(__bridge CFStringRef)capture);
        CFDictionarySetValue(input,CFSTR("OutputFileName"),CFSTR("model"));
        CFDictionarySetValue(input,CFSTR("OutputFilePath"),(__bridge CFStringRef)[output stringByAppendingString:@"/"]);
        NSDictionary *options=@{@"TargetArchitecture":architecture};
        dispatch_semaphore_t done=dispatch_semaphore_create(0);
        __block int status=INT_MIN;
        compile(input,(__bridge CFDictionaryRef)options,^(int code,CFDictionaryRef details) {
            status=code;
            if (code) fprintf(stderr,"ANECCompile=%d details=%s\n",code,
                              [(__bridge NSDictionary *)details description].UTF8String);
            dispatch_semaphore_signal(done);
        });
        long timedOut=dispatch_semaphore_wait(done,dispatch_time(DISPATCH_TIME_NOW,300*NSEC_PER_SEC));
        CFRelease(input);CFRelease(base);
        if (timedOut || status) {
            fprintf(stderr,"Apple compile status=%d timeout=%ld\n",status,timedOut);
            return 5;
        }
        NSString *destination=[output stringByAppendingPathComponent:@"model.hwx"];
        for (NSString *name in [files enumeratorAtPath:output]) {
            NSString *path=[output stringByAppendingPathComponent:name];
            NSData *data=[NSData dataWithContentsOfFile:path options:NSDataReadingMappedIfSafe error:NULL];
            if (data.length<4) continue;
            const unsigned char *bytes=data.bytes;
            if (bytes[0]!=0xce || bytes[1]!=0xfa || bytes[2]!=0xef || bytes[3]!=0xbe) continue;
            if ([path isEqualToString:destination]) return 0;
            return [data writeToFile:destination options:NSDataWritingAtomic error:&error]?0:6;
        }
        fprintf(stderr,"compiler returned no HWX object\n");
        return 7;
    }
}
