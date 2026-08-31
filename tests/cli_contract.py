"""Standard-library-only integration tests for scaffold CLI contracts."""

import json
import subprocess
import sys


def run(executable, *arguments):
    return subprocess.run(
        [executable, *arguments], text=True, capture_output=True, timeout=20, check=False
    )


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    mode, executable = sys.argv[1:]
    if mode == "device":
        result = run(executable, "--json")
        require(result.returncode == 0, result.stderr)
        data = json.loads(result.stdout)
        require(data["schema_version"] == 1, "Unexpected device schema")
        require(isinstance(data["cuda_compiled"], bool), "CUDA flag must be boolean")
        require(data["status"] in {"disabled", "unavailable", "ready"}, "Unexpected status")
        require(isinstance(data["devices"], list), "Device list required")
        require((data["status"] == "ready") == bool(data["devices"]), "Status/device mismatch")
        if not data["cuda_compiled"]:
            require(data["status"] == "disabled", "CPU-only status must be disabled")
        for device in data["devices"]:
            require(device["compute_major"] > 0, "Invalid compute capability")
            require(device["total_memory_bytes"] > 0, "Invalid memory capacity")
        strict = run(executable, "--json", "--require-gpu")
        expected = 0 if data["devices"] else 2
        require(strict.returncode == expected, "Strict device exit code mismatch")
        json.loads(strict.stdout)
    elif mode == "model":
        result = run(executable, "--print-config")
        require(result.returncode == 0, result.stderr)
        config = json.loads(result.stdout)
        require(config["hidden_size"] == 512, "Default model shape changed")
        require(config["num_layers"] == 4, "Default layer count changed")
        require(config["head_dim"] * config["num_heads"] == config["hidden_size"], "Invalid heads")
        require(run(executable, "--generate").returncode == 1, "Unimplemented inference accepted")
    elif mode == "bench":
        result = run(executable, "--list")
        require(result.returncode == 0, result.stderr)
        require("No benchmark cases" in result.stdout, "Scaffold must not claim real benchmarks")
        require(run(executable, "--run", "gemm").returncode == 1, "Unimplemented benchmark accepted")
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
