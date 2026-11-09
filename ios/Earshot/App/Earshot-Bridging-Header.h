// Earshot-Bridging-Header.h - the C headers Swift can see (30-ios 1).
// earshot.h is the engine API (contract 4); AudioHost.h is the RemoteIO host (contract 5).
#include "earshot.h"
#include "AudioHost.h"
#include "MallocHook.h"   // I-1: the debug-only malloc counter (compiled out of Release)
