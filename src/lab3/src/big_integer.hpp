#pragma once

#include <algorithm>
#include <cstdint>
#include <format>
#include <string>
#include <vector>

class BigInteger final {
public:
	void add(std::int64_t value) {
		if (value == 0) return;

		BigInteger other;
		other.negative_ = value < 0;
		std::uint64_t magnitude = 0;
		if (value < 0) {
			magnitude = static_cast<std::uint64_t>(-(value + 1)) + 1;
		}
		else {
			magnitude = static_cast<std::uint64_t>(value);
		}

		while (magnitude > 0) {
			other.digits_.push_back(static_cast<std::uint32_t>(magnitude % base_));
			magnitude /= base_;
		}

		add(other);
	}

	void add(const BigInteger& other) {
		if (other.digits_.empty()) return;
		if (digits_.empty()) {
			*this = other;
			return;
		}

		if (negative_ == other.negative_) {
			add_abs(other);
			return;
		}

		const int cmp = compare_abs(other);
		if (cmp == 0) {
			digits_.clear();
			negative_ = false;
		}
		else if (cmp > 0) {
			sub_abs(other);
		}
		else {
			BigInteger tmp = other;
			tmp.sub_abs(*this);
			*this = std::move(tmp);
		}
	}

	std::string to_string() const {
		if (digits_.empty()) return "0";

		std::string result;
		if (negative_) result.push_back('-');
		result += std::to_string(digits_.back());
		for (auto it = digits_.rbegin() + 1; it != digits_.rend(); ++it) {
			result += std::format("{:09}", *it);
		}
		return result;
	}

private:
	static constexpr std::uint32_t base_ = 1'000'000'000;

	bool negative_ = false;
	std::vector<std::uint32_t> digits_;

	void normalize() {
		while (!digits_.empty() && digits_.back() == 0) {
			digits_.pop_back();
		}
		if (digits_.empty()) negative_ = false;
	}

	int compare_abs(const BigInteger& other) const {
		if (digits_.size() != other.digits_.size()) {
			return digits_.size() < other.digits_.size() ? -1 : 1;
		}

		for (std::size_t i = digits_.size(); i-- > 0;) {
			if (digits_[i] != other.digits_[i]) {
				return digits_[i] < other.digits_[i] ? -1 : 1;
			}
		}
		return 0;
	}

	void add_abs(const BigInteger& other) {
		const std::size_t n = std::max(digits_.size(), other.digits_.size());
		digits_.resize(n, 0);

		std::uint64_t carry = 0;
		for (std::size_t i = 0; i < n; ++i) {
			const std::uint64_t rhs = i < other.digits_.size() ? other.digits_[i] : 0;
			const std::uint64_t sum = static_cast<std::uint64_t>(digits_[i]) + rhs + carry;
			digits_[i] = static_cast<std::uint32_t>(sum % base_);
			carry = sum / base_;
		}

		if (carry) digits_.push_back(carry);
	}

	void sub_abs(const BigInteger& other) {
		std::int64_t borrow = 0;
		for (std::size_t i = 0; i < digits_.size(); ++i) {
			const std::int64_t rhs = i < other.digits_.size() ? other.digits_[i] : 0;
			std::int64_t current = static_cast<std::int64_t>(digits_[i]) - rhs - borrow;
			if (current < 0) {
				current += base_;
				borrow = 1;
			}
			else {
				borrow = 0;
			}
			digits_[i] = static_cast<std::uint32_t>(current);
		}

		normalize();
	}
};
