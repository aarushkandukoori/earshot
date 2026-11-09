# Earshot engine Makefile (20-engine 2.1 and 2.2).
#
#   make            build build/libengine.a (host, tests only), build/esrun and every test binary
#   make test       run the unit tests (and the other tests once their sources exist)
#   make test SAN=1 the same with address and undefined-behavior sanitizers (test_units, test_parity)
#                   (on this Mac, macOS 26.5 + Xcode clang 17, ASan hangs at startup even for an empty
#                   program; UBSan alone works: make test SAN=1 SAN_FLAGS="-fsanitize=undefined")
#   make parity     test_parity against $(EXPORT)            (needs the ML export, package E-1)
#   make fuzz       one simulated hour of hostile input under nice (FUZZ_SECONDS, FUZZ_SEED, FUZZ_ARGS)
#   make bench      per-hop timing, JSON lines on stdout (P-cores; make bench-ecore for the E-cores)
#   make bench-xt   the same on the opt-in -DES_HOP_SGEMM_XT build (build/xt), for the hop-path A/B
#   make bench-save for the lead on a quiet Mac: waits for ../bench/wait_quiet.sh, then appends the
#                   synthetic and $(EXPORT) rows (P-cores and E-cores) plus the $(EXPORT) xt rows
#                   (P-cores) to bench-results/YYYY-MM-DD.jsonl, only if every run succeeded
#                   (NOTE="..." adds a note field; BENCH_WAIT=0 skips the wait)
#   make test-xt    units, rt (10,000 hops, guarded) and parity on the opt-in -DES_HOP_SGEMM_XT build
#   make sim-rt     test_rt (guarded, 10,000 hops) on the booted iOS simulator, for the shipping build
#                   and the -DES_HOP_SGEMM_XT build: the simulator runs iOS's Accelerate, which macOS
#                   tests cannot see (needs a booted simulator; it is not the iPhone either)
#   make rt         test_rt with 100,000 hops and 3 instances under nice (RT_HOPS=... to change)
#   make latency    test_latency alone (impulse delay, alignment, gain cap, hand-off, slack step)
#   make esrun      build the offline WAV tool (build/esrun, 20-engine 14.7)
#   make ios-check  compile every src/*.c for arm64-apple-ios26.0 (no link), with the app's defines
#                   (IOS_APP_DEFS, as ios/project.yml) and again with -DES_HOP_SGEMM_XT
#
# Hop path (E-3 fix pass, nn.c es_linear_hop): the default build uses one cblas_sgemv per row for the
# hop's 16-row products, which is what the app ships (ios/project.yml also defines ES_HOP_SGEMV=1).
# -DES_HOP_SGEMM_XT selects the faster transposed sgemm; it stays opt-in until the lead's D68 malloc
# hook shows 0 allocations on the iPhone. make test, rt, parity and bench test the default build.
#   make clean      remove build/
#
# Targets whose sources do not exist yet print "not built yet" and succeed.
# Never add -ffast-math, -Ofast, -ffinite-math-only or -funsafe-math-optimizations (20-engine 2.1).

CC      = xcrun clang
BASE_CFLAGS = -std=c11 -O2 -g -Wall -Wextra -Wshadow -Wstrict-prototypes -Werror \
              -DACCELERATE_NEW_LAPACK -Iinclude -Isrc
CFLAGS  = $(BASE_CFLAGS)
LDLIBS  = -framework Accelerate          # CommonCrypto is part of libSystem
SAN_FLAGS = -fsanitize=address,undefined -fno-omit-frame-pointer
EXPORT ?= ../ml/exports/random_v1

IOS_CC     = xcrun --sdk iphoneos clang
IOS_TARGET = -target arm64-apple-ios26.0
# The app's defines, copied from ios/project.yml GCC_PREPROCESSOR_DEFINITIONS (keep the two in step).
IOS_APP_DEFS = -DACCELERATE_NEW_LAPACK=1 -DES_HOP_SGEMV=1

# The opt-in hop path builds into its own folder (test-xt, bench-xt, bench-save).
XT_BUILD  = build/xt
XT_CFLAGS = $(BASE_CFLAGS) -DES_HOP_SGEMM_XT

# sim-rt: iOS simulator builds (arm64 simulator slice of iOS 26.0) in build/sim.
SIM_CC     = xcrun --sdk iphonesimulator clang
SIM_TARGET = -target arm64-apple-ios26.0-simulator
SIM_BUILD  = build/sim

BUILD = build
SRCS  = $(wildcard src/*.c)
HDRS  = $(wildcard src/*.h) include/earshot.h
OBJS  = $(patsubst src/%.c,$(BUILD)/obj/%.o,$(SRCS))

# Sanitizer builds go to their own folder so they never mix with the normal objects.
ifeq ($(SAN),1)
  BUILD := build/san
  OBJS  := $(patsubst src/%.c,$(BUILD)/obj/%.o,$(SRCS))
  CFLAGS += $(SAN_FLAGS)
  LDFLAGS += $(SAN_FLAGS)
endif

# Test and tool binaries that exist so far (each is built only when its source file exists).
HAVE_PARITY  = $(wildcard tests/test_parity.c)
HAVE_RT      = $(wildcard tests/test_rt.c)
HAVE_FUZZ    = $(wildcard tests/test_fuzz.c)
HAVE_LATENCY = $(wildcard tests/test_latency.c)
HAVE_BENCH   = $(wildcard tests/bench.c)
HAVE_ESRUN   = $(wildcard tools/esrun.c)

BINS = $(BUILD)/test_units
ifneq ($(HAVE_PARITY),)
  BINS += $(BUILD)/test_parity
endif
ifneq ($(HAVE_RT),)
  BINS += $(BUILD)/test_rt $(BUILD)/rt_guard.dylib
endif
ifneq ($(HAVE_FUZZ),)
  BINS += $(BUILD)/test_fuzz
endif
ifneq ($(HAVE_LATENCY),)
  BINS += $(BUILD)/test_latency
endif
ifneq ($(HAVE_BENCH),)
  BINS += $(BUILD)/bench
endif
ifneq ($(HAVE_ESRUN),)
  BINS += $(BUILD)/esrun
endif

# A sanitizer build makes only the two tests that use sanitizers.
ifeq ($(SAN),1)
  BINS := $(BUILD)/test_units $(if $(HAVE_PARITY),$(BUILD)/test_parity)
endif

.PHONY: all test parity fuzz bench bench-ecore bench-xt bench-save esrun ios-check clean rt latency test-xt sim-rt

all: $(BUILD)/libengine.a $(BINS)

# ---------------------------------------------------------------------------------------------
# Library and binaries
# ---------------------------------------------------------------------------------------------
$(BUILD)/obj/%.o: src/%.c $(HDRS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/libengine.a: $(OBJS)
	@mkdir -p $(dir $@)
	rm -f $@
	ar rcs $@ $(OBJS)

# test_units also compiles the WAV reader and writer (tools/wav.c) for its WAV rows.
$(BUILD)/test_units: tests/test_units.c tools/wav.c tools/wav.h $(BUILD)/libengine.a $(HDRS)
	$(CC) $(CFLAGS) $(LDFLAGS) -Itools tests/test_units.c tools/wav.c $(BUILD)/libengine.a $(LDLIBS) -o $@

$(BUILD)/test_parity: tests/test_parity.c $(BUILD)/libengine.a $(HDRS)
	$(CC) $(CFLAGS) $(LDFLAGS) tests/test_parity.c $(BUILD)/libengine.a $(LDLIBS) -o $@

$(BUILD)/test_rt: tests/test_rt.c $(BUILD)/libengine.a $(HDRS)
	$(CC) $(CFLAGS) tests/test_rt.c $(BUILD)/libengine.a $(LDLIBS) -o $@

$(BUILD)/rt_guard.dylib: tests/rt_guard.c
	$(CC) $(CFLAGS) -dynamiclib tests/rt_guard.c -o $@

$(BUILD)/test_fuzz: tests/test_fuzz.c $(BUILD)/libengine.a $(HDRS)
	$(CC) $(CFLAGS) $(LDFLAGS) tests/test_fuzz.c $(BUILD)/libengine.a $(LDLIBS) -o $@

$(BUILD)/test_latency: tests/test_latency.c $(BUILD)/libengine.a $(HDRS)
	$(CC) $(CFLAGS) $(LDFLAGS) tests/test_latency.c $(BUILD)/libengine.a $(LDLIBS) -o $@

$(BUILD)/bench: tests/bench.c $(BUILD)/libengine.a $(HDRS)
	$(CC) $(CFLAGS) tests/bench.c $(BUILD)/libengine.a $(LDLIBS) -o $@

$(BUILD)/esrun: tools/esrun.c tools/wav.c tools/wav.h $(BUILD)/libengine.a $(HDRS)
	$(CC) $(CFLAGS) $(LDFLAGS) -Itools tools/esrun.c tools/wav.c $(BUILD)/libengine.a $(LDLIBS) -o $@

# ---------------------------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------------------------
# test: units always; latency, rt (10,000 hops), fuzz (60 s) once they exist; parity if the export
# exists. With SAN=1 only test_units and test_parity run (test_rt and bench never use sanitizers).
# Parity always runs with single-threaded BLAS (VECLIB_MAXIMUM_THREADS=1, 20-engine 3.5) so its
# run-twice memcmp determinism check is meaningful.
test: all
	$(BUILD)/test_units
ifneq ($(SAN),1)
ifneq ($(HAVE_LATENCY),)
	$(BUILD)/test_latency
else
	@echo "test_latency: not built yet (package E-2)"
endif
ifneq ($(HAVE_RT),)
	DYLD_INSERT_LIBRARIES=$(BUILD)/rt_guard.dylib RT_HOPS=10000 $(BUILD)/test_rt
else
	@echo "test_rt: not built yet (package E-2)"
endif
ifneq ($(HAVE_FUZZ),)
	FUZZ_SECONDS=60 $(BUILD)/test_fuzz
else
	@echo "test_fuzz: not built yet (package E-2)"
endif
endif
ifneq ($(HAVE_PARITY),)
	@if [ -d "$(EXPORT)" ]; then VECLIB_MAXIMUM_THREADS=1 $(BUILD)/test_parity $(EXPORT); \
	 else echo "test_parity: skipped, $(EXPORT) does not exist yet"; fi
else
	@echo "test_parity: not built yet (package E-1)"
endif

parity: all
ifneq ($(HAVE_PARITY),)
	VECLIB_MAXIMUM_THREADS=1 $(BUILD)/test_parity $(EXPORT)
else
	@echo "parity: not built yet (package E-1)"
endif

# test-xt: the opt-in hop path (-DES_HOP_SGEMM_XT) in build/xt, so it keeps passing while it is not the
# default: units, rt (10,000 hops with the allocation guard, macOS) and parity if the export exists.
test-xt:
	$(MAKE) all BUILD=$(XT_BUILD) CFLAGS='$(XT_CFLAGS)'
	$(XT_BUILD)/test_units
	DYLD_INSERT_LIBRARIES=$(XT_BUILD)/rt_guard.dylib RT_HOPS=10000 $(XT_BUILD)/test_rt
	@if [ -d "$(EXPORT)" ]; then VECLIB_MAXIMUM_THREADS=1 $(XT_BUILD)/test_parity $(EXPORT); \
	 else echo "test_parity: skipped, $(EXPORT) does not exist yet"; fi

# sim-rt: test_rt on the booted iOS simulator (xcrun simctl spawn booted). The E-3 review found that
# iOS's Accelerate allocates where macOS's does not (transposed sgemm with 512 columns), so this is the
# closest check before the phone. Two builds: the app's defines (IOS_APP_DEFS) and -DES_HOP_SGEMM_XT.
# simctl passes SIMCTL_CHILD_* variables to the process without the prefix, which is how the guard
# (DYLD_INSERT_LIBRARIES) and RT_HOPS reach it. Paths are absolute: the process does not start in engine/.
# With real weights when $(EXPORT) exists, else the synthetic model.
SIM_HOPS ?= 10000
SIM_EXPORT_ARG = $(if $(wildcard $(EXPORT)/earshot.esw),--export $(abspath $(EXPORT)))
sim-rt:
	@xcrun simctl list devices booted | grep -q Booted || { echo "sim-rt: no booted simulator (boot one first)"; exit 1; }
	@mkdir -p $(SIM_BUILD)
	$(SIM_CC) $(SIM_TARGET) $(BASE_CFLAGS) $(IOS_APP_DEFS) $(SRCS) tests/test_rt.c $(LDLIBS) -o $(SIM_BUILD)/test_rt
	$(SIM_CC) $(SIM_TARGET) $(BASE_CFLAGS) -DES_HOP_SGEMM_XT $(SRCS) tests/test_rt.c $(LDLIBS) -o $(SIM_BUILD)/test_rt_xt
	$(SIM_CC) $(SIM_TARGET) $(BASE_CFLAGS) -dynamiclib tests/rt_guard.c -o $(SIM_BUILD)/rt_guard.dylib
	SIMCTL_CHILD_DYLD_INSERT_LIBRARIES=$(abspath $(SIM_BUILD))/rt_guard.dylib SIMCTL_CHILD_RT_HOPS=$(SIM_HOPS) \
	  nice -n 15 xcrun simctl spawn booted $(abspath $(SIM_BUILD))/test_rt $(SIM_EXPORT_ARG)
	SIMCTL_CHILD_DYLD_INSERT_LIBRARIES=$(abspath $(SIM_BUILD))/rt_guard.dylib SIMCTL_CHILD_RT_HOPS=$(SIM_HOPS) \
	  nice -n 15 xcrun simctl spawn booted $(abspath $(SIM_BUILD))/test_rt_xt $(SIM_EXPORT_ARG)

# rt: the full 20-engine 14.3 run (100,000 hops, 3 instances; minutes of CPU, so under nice).
# DYLD_INSERT_LIBRARIES goes through `env`: macOS strips DYLD_* variables when it starts a protected
# system binary such as /usr/bin/nice, so they must be set by a command that nice starts.
RT_HOPS ?= 100000
rt: all
	nice -n 15 env DYLD_INSERT_LIBRARIES=$(BUILD)/rt_guard.dylib RT_HOPS=$(RT_HOPS) $(BUILD)/test_rt

latency: all
	$(BUILD)/test_latency

# fuzz: one simulated hour (20-engine 14.4). Many minutes of CPU, so under nice; FUZZ_SEED=... replays a
# seed, FUZZ_ARGS="--export ../ml/exports/random_v1" uses real weights instead of the synthetic model.
FUZZ_SECONDS ?= 3600
fuzz: all
ifneq ($(HAVE_FUZZ),)
	nice -n 15 env FUZZ_SECONDS=$(FUZZ_SECONDS) $(BUILD)/test_fuzz $(FUZZ_ARGS)
else
	@echo "fuzz: not built yet (package E-2)"
endif

# bench: 20-engine 14.6. Single-threaded BLAS; a missing $(EXPORT) falls back to the synthetic model.
# Timings are only meaningful on a quiet machine (bench-save waits for one).
bench: all
	VECLIB_MAXIMUM_THREADS=1 $(BUILD)/bench $(EXPORT)

bench-ecore: all
	VECLIB_MAXIMUM_THREADS=1 $(BUILD)/bench --ecore $(EXPORT)

bench-xt:
	$(MAKE) all BUILD=$(XT_BUILD) CFLAGS='$(XT_CFLAGS)'
	VECLIB_MAXIMUM_THREADS=1 $(XT_BUILD)/bench $(EXPORT)

# bench-save: the rerun target for the lead. ../bench/wait_quiet.sh (load below 4 and GPU below 15%,
# up to BENCH_WAIT seconds, then it continues anyway and says so), then 5 runs: synthetic and $(EXPORT)
# on the shipping build, each on the P-cores and under taskpolicy -c background (rows named *_ecore), and
# $(EXPORT) on the xt build (P-cores) for the hop-path A/B. Every row carries its own load_start and
# hop_path. The runs write to $(BENCH_TMP) first (the first one truncates it); make stops at the first
# failing run, so the day's file gets the rows only when all 5 succeeded.
# Example: make bench-save NOTE="quiet Mac, training stopped"
BENCH_WAIT ?= 900
BENCH_DAY  ?= $(shell date +%F)
BENCH_NOTE_ARG = $(if $(NOTE),--note "$(NOTE)")
BENCH_TMP  = $(BUILD)/bench-save.tmp.jsonl
bench-save: all
	-../bench/wait_quiet.sh 4 $(BENCH_WAIT)
	$(MAKE) all BUILD=$(XT_BUILD) CFLAGS='$(XT_CFLAGS)'
	@mkdir -p bench-results
	VECLIB_MAXIMUM_THREADS=1 $(BUILD)/bench $(BENCH_NOTE_ARG) synthetic > $(BENCH_TMP)
	VECLIB_MAXIMUM_THREADS=1 $(BUILD)/bench $(BENCH_NOTE_ARG) $(EXPORT) >> $(BENCH_TMP)
	VECLIB_MAXIMUM_THREADS=1 $(BUILD)/bench --ecore $(BENCH_NOTE_ARG) synthetic >> $(BENCH_TMP)
	VECLIB_MAXIMUM_THREADS=1 $(BUILD)/bench --ecore $(BENCH_NOTE_ARG) $(EXPORT) >> $(BENCH_TMP)
	VECLIB_MAXIMUM_THREADS=1 $(XT_BUILD)/bench $(BENCH_NOTE_ARG) $(EXPORT) >> $(BENCH_TMP)
	cat $(BENCH_TMP) >> bench-results/$(BENCH_DAY).jsonl
	@echo "bench-save: appended the rows of $(BENCH_TMP) to bench-results/$(BENCH_DAY).jsonl"

esrun: $(BUILD)/libengine.a
ifneq ($(HAVE_ESRUN),)
	$(MAKE) $(BUILD)/esrun
else
	@echo "esrun: not built yet (package E-2)"
endif

# Compile every engine source for the phone (no link, no sanitizers): once with the app's defines
# (IOS_APP_DEFS, the shipping hop path) and once with the opt-in -DES_HOP_SGEMM_XT.
ios-check:
	@for defs in "$(IOS_APP_DEFS)" "-DES_HOP_SGEMM_XT"; do \
	  for f in $(SRCS); do \
	    echo "ios-check [$$defs] $$f"; \
	    $(IOS_CC) $(IOS_TARGET) $(BASE_CFLAGS) $$defs -c $$f -o /dev/null || exit 1; \
	  done; \
	done
	@echo "ios-check: all sources compile for arm64-apple-ios26.0 (app defines and ES_HOP_SGEMM_XT)"

clean:
	rm -rf build
