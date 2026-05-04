#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>

namespace fs = std::filesystem;

namespace {

constexpr std::array<std::pair<std::string, std::uint64_t>, 5> preset{ {
	{ "1M", 1ULL << 20 },
	{ "4M", 4ULL << 20 },
	{ "16M", 16ULL << 20 },
	{ "64M", 64ULL << 20 },
	{ "128M", 128ULL << 20 },
} };

void gen_array(const fs::path& path, std::size_t size, std::mt19937_64& e) {
	static_assert(sizeof(std::int32_t) == 4);

	std::ofstream out(path, std::ios::binary);
	if (!out.is_open()) {
		throw std::runtime_error("Failed to open array output file: " + path.string());
	}

	std::uniform_int_distribution<std::int32_t> u(std::numeric_limits<std::int32_t>::min(),
											 std::numeric_limits<std::int32_t>::max());
	for (std::size_t i = 0; i < size; ++i) {
		const std::int32_t value = u(e);
		out.write(reinterpret_cast<const char*>(&value), sizeof(value));
	}
}

} // namespace

int main() {
	fs::create_directories("data/array");

	std::mt19937_64 e(
		static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count()));
	for (const auto& [label, size] : preset) {
		gen_array(fs::path("data/array") / std::format("A_{}.in", label), size, e);
	}

	return 0;
}
