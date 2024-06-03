.PHONY: check deps fmt test

# Define a pattern rule for executing scripts
define SCRIPT_RULE
$1:
	bash scripts/$1.sh
endef

$(foreach target,check deps fmt test,$(eval $(call SCRIPT_RULE,$(target))))
