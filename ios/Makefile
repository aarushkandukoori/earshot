# Earshot iOS app (30-ios 2.2). Run from ios/: make project, make build, make test.
# ENGINE is real since lead integration (contract 11): the app compiles ../engine/src/*.c directly
# (20-engine 2.3). ENGINE=stub switches back to Stubs/earshot_stub.c, for the tests that only make
# sense with the stub's pass-through behavior: make test-stub (same as make test ENGINE=stub).
# Personal settings live in ios/local.mk (gitignored, optional), for example:
#   DEVICE ?= <your iPhone's UDID, from: make devices>
#   TEAM   ?= <your Apple development team id, from Xcode > Settings > Accounts>
# Each can also come from the environment or the command line (make install DEVICE=... TEAM=...).
# DEVICE can also be saved once with: make devices; echo <udid> > .device
-include local.mk
ENGINE ?= real
SRC     = $(if $(filter real,$(ENGINE)),../engine/src,Stubs)
DEVICE ?= $(shell cat .device 2>/dev/null)
BUNDLE  = com.aarushkandukoori.earshot
CONFIG ?= Release
APP     = build/Build/Products/$(CONFIG)-iphoneos/Earshot.app
XC      = xcodebuild -project Earshot.xcodeproj -scheme Earshot -derivedDataPath build
XCODEGEN ?= /opt/homebrew/bin/xcodegen
# TEAM is the signing team. `make project` writes it into the generated project (project.yml reads
# EARSHOT_TEAM), and device builds also pass it to xcodebuild. A free Personal Team works (DECISIONS L
# 2026-11-07 "Signing resolved 23:14"); profiles from a free team last 7 days.
TEAM    ?=
TEAMSET  = $(if $(TEAM),DEVELOPMENT_TEAM=$(TEAM))

.PHONY: project build build-sim build-unsigned models install test test-stub test-device devices logs phase0 privacy-check

project:      ; EARSHOT_ENGINE_SRC=$(SRC) EARSHOT_TEAM=$(TEAM) $(XCODEGEN) generate --spec project.yml
build: project; $(XC) -configuration $(CONFIG) -destination 'generic/platform=iOS' -allowProvisioningUpdates $(TEAMSET) build
build-sim: project; $(XC) -configuration $(CONFIG) -destination 'generic/platform=iOS Simulator' build
# Proves the device build compiles without the owner's Xcode account (signing needs it, DECISIONS L)
build-unsigned: project; $(XC) -configuration $(CONFIG) -destination 'generic/platform=iOS' CODE_SIGNING_ALLOWED=NO build
# Copies the ML exports into Earshot/Resources/Models (the root Makefile's only target)
models:       ; $(MAKE) -C .. models
install: build
	xcrun devicectl device install app --device $(DEVICE) $(APP)
	xcrun devicectl device process launch --device $(DEVICE) $(BUNDLE)
test: project ; $(XC) -destination 'platform=iOS Simulator,name=iPhone 17 Pro,OS=26.0' test
# The whole suite against the stub engine (regenerates the project with Stubs/; run make project after)
test-stub:    ; $(MAKE) test ENGINE=stub
test-device: project ; $(XC) -destination 'id=$(DEVICE)' -allowProvisioningUpdates $(TEAMSET) test
devices:      ; xcrun devicectl list devices
logs:         ; xcrun devicectl device copy from --device $(DEVICE) --domain-type appDataContainer \
                  --domain-identifier $(BUNDLE) --source Documents/logs --destination build/logs
phase0:       ; xcrun devicectl device copy from --device $(DEVICE) --domain-type appDataContainer \
                  --domain-identifier $(BUNDLE) --source Documents/phase0 --destination ../docs/measurements
privacy-check: ; ! grep -rnE 'URLSession|NWConnection|import Network|WKWebView' Earshot
