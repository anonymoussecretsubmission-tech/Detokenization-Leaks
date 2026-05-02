# Makefile for Data Collection Phase

# --- Paths ---
# Adjust these if your CacheSC location changes
CACHESC_INCLUDE := /home/user/Projects/CacheSC/include
CACHESC_LIBDIR  := /home/user/Projects/CacheSC/lib

SRC_DIR         := ./
BIN_DIR         := ../bin

# --- Compiler Settings ---
CC      := gcc
CFLAGS  := -g -O1 -fno-tree-vectorize -std=gnu99 \
           -I$(CACHESC_INCLUDE)

LDFLAGS := -L$(CACHESC_LIBDIR)
# Linking Mastik and CacheSC
LDLIBS  := -lcachesc -ldwarf -lelf -lbfd -lpthread -lmastik

# --- Targets ---
TARGET := $(BIN_DIR)/collect_attacker

.PHONY: all clean

all: $(TARGET)

# Ensure the output directory exists
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Compilation Rule
$(TARGET): $(SRC_DIR)/collect_attacker.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)
	@echo "Build successful! Target: $@"

clean:
	rm -f $(TARGET)