# Earshot root Makefile. For now it has one target, delegated to WP-IOS by the lead:
#
#   make models            copies the ML exports the iOS app bundles into ios/Earshot/Resources/Models/
#   make models EXPORT=v1  same, from ml/exports/v1 once the trained model exists
#
# What lands in ios/Earshot/Resources/Models/ (gitignored by ios/.gitignore):
#   everything in ml/exports/$(EXPORT)/  (earshot.esw, testvec_short.est, testvec_long.est,
#                                         thresholds.json, wespk_selftest.est, quality_vectors.json)
#   wespeaker_resnet34_fp32.mlpackage    (Xcode compiles it to .mlmodelc inside the app)
# ios/project.yml keeps quality_vectors.json and testvec_long.est out of the app (the test bundle
# takes quality_vectors.json), so the app holds exactly the 5 files of 30-ios 1.
#
# ml/exports is read-only here: this only copies from it.

EXPORT     ?= random_v1
EXPORT_DIR  = ml/exports/$(EXPORT)
WESPEAKER   = ml/exports/wespeaker_resnet34_fp32.mlpackage
MODELS_DIR  = ios/Earshot/Resources/Models

.PHONY: models

models:
	@test -f $(EXPORT_DIR)/earshot.esw || { echo "missing $(EXPORT_DIR)/earshot.esw (run the ML export first)"; exit 1; }
	@test -d $(WESPEAKER) || { echo "missing $(WESPEAKER)"; exit 1; }
	mkdir -p $(MODELS_DIR)
	cp -p $(EXPORT_DIR)/* $(MODELS_DIR)/
	# rsync --delete keeps the package an exact copy (no stale files from an older export)
	rsync -a --delete $(WESPEAKER)/ $(MODELS_DIR)/wespeaker_resnet34_fp32.mlpackage/
	@ls -l $(MODELS_DIR)
