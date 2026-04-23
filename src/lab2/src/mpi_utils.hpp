#pragma once

#include <array>
#include <cstddef>
#include <format>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <mpi.h>

void mpi_check(int code, const char* call) {
	if (code == MPI_SUCCESS) return;

	static char message[MPI_MAX_ERROR_STRING]{};
	int message_len = 0;
	MPI_Error_string(code, message, &message_len);
	throw std::runtime_error(
		std::format("MPI call failed: {} -> {}", call, std::string(message, message_len)));
}

template <typename T>
struct MpiType;

template <>
struct MpiType<int> {
	static constexpr MPI_Datatype value = MPI_INT;
};

template <>
struct MpiType<float> {
	static constexpr MPI_Datatype value = MPI_FLOAT;
};

template <>
struct MpiType<double> {
	static constexpr MPI_Datatype value = MPI_DOUBLE;
};

template <typename T>
constexpr MPI_Datatype mpi_type_v = MpiType<T>::value;

class MpiDatatype final {
public:
	MpiDatatype() = default;

	explicit MpiDatatype(MPI_Datatype type)
		: type_(type) {
	}

	~MpiDatatype() {
		reset();
	}

	MpiDatatype(const MpiDatatype&) = delete;
	MpiDatatype& operator=(const MpiDatatype&) = delete;

	MpiDatatype(MpiDatatype&& other) noexcept
		: type_(std::exchange(other.type_, MPI_DATATYPE_NULL)) {
	}

	MpiDatatype& operator=(MpiDatatype&& other) noexcept {
		if (this != &other) {
			reset();
			type_ = std::exchange(other.type_, MPI_DATATYPE_NULL);
		}
		return *this;
	}

	MPI_Datatype get() const noexcept {
		return type_;
	}

	void reset() {
		if (type_ != MPI_DATATYPE_NULL) {
			MPI_Datatype type = type_;
			type_ = MPI_DATATYPE_NULL;
			mpi_check(MPI_Type_free(&type), "MPI_Type_free");
		}
	}

private:
	MPI_Datatype type_ = MPI_DATATYPE_NULL;
};

template <std::size_t N>
MpiDatatype make_struct_datatype(const std::array<int, N>& block_lengths,
								 const std::array<MPI_Aint, N>& displacements,
								 const std::array<MPI_Datatype, N>& types) {
	MPI_Datatype type = MPI_DATATYPE_NULL;
	mpi_check(MPI_Type_create_struct(int(N), block_lengths.data(),
									 displacements.data(), types.data(), &type),
			  "MPI_Type_create_struct");
	mpi_check(MPI_Type_commit(&type), "MPI_Type_commit");
	return MpiDatatype(type);
}

class MPIEnvironment final {
public:
	MPIEnvironment(int& argc, char**& argv) {
		mpi_check(MPI_Init(&argc, &argv), "MPI_Init");
	}

	~MPIEnvironment() noexcept {
		int finalized = 0;
		MPI_Finalized(&finalized);
		if (!finalized) {
			MPI_Finalize();
		}
	}

	MPIEnvironment(const MPIEnvironment&) = delete;
	MPIEnvironment& operator=(const MPIEnvironment&) = delete;
	MPIEnvironment(MPIEnvironment&&) = delete;
	MPIEnvironment& operator=(MPIEnvironment&&) = delete;
};

class MPIWorld final {
public:
	explicit MPIWorld(MPI_Comm comm = MPI_COMM_WORLD)
		: comm_(comm) {
		mpi_check(MPI_Comm_rank(comm_, &rank_), "MPI_Comm_rank");
		mpi_check(MPI_Comm_size(comm_, &size_), "MPI_Comm_size");
	}

	int rank() const noexcept {
		return rank_;
	}

	int size() const noexcept {
		return size_;
	}

	void barrier() const {
		mpi_check(MPI_Barrier(comm_), "MPI_Barrier");
	}

	template <typename T>
	void send(const std::vector<T>& data, int dest, int tag = 0) const {
		mpi_check(
			MPI_Send(data.data(), int(data.size()), mpi_type_v<T>, dest, tag,
					 comm_),
			"MPI_Send");
	}

	template <typename T>
	void recv(std::vector<T>& data, int src, int tag = 0) const {
		mpi_check(
			MPI_Recv(data.data(), int(data.size()), mpi_type_v<T>, src, tag,
					 comm_, MPI_STATUS_IGNORE),
			"MPI_Recv");
	}

	template <typename T>
	void bcast(std::vector<T>& data, int root = 0) const {
		mpi_check(
			MPI_Bcast(data.data(), int(data.size()), mpi_type_v<T>, root, comm_),
			"MPI_Bcast");
	}

	template <typename T>
	void bcast_value(T& value, MPI_Datatype type, int root = 0) const {
		mpi_check(MPI_Bcast(&value, 1, type, root, comm_), "MPI_Bcast");
	}

	template <typename T>
	void scatter_equal(const std::vector<T>& send,
					   std::vector<T>& recv,
					   int root = 0) const {
		const T* send_ptr = rank_ == root ? send.data() : nullptr;
		mpi_check(
			MPI_Scatter(send_ptr, int(recv.size()), mpi_type_v<T>, recv.data(),
						int(recv.size()), mpi_type_v<T>, root, comm_),
			"MPI_Scatter");
	}

	template <typename T>
	void gather_equal(const std::vector<T>& send,
					  std::vector<T>& recv,
					  int root = 0) const {
		T* recv_ptr = rank_ == root ? recv.data() : nullptr;
		mpi_check(
			MPI_Gather(send.data(), send.size(), mpi_type_v<T>, recv_ptr,
					   recv.size(), mpi_type_v<T>, root, comm_),
			"MPI_Gather");
	}

	template <typename T>
	void scatterv(const std::vector<T>& send,
				  const std::vector<int>& send_counts,
				  const std::vector<int>& displacements,
				  std::vector<T>& recv,
				  int root = 0) const {
		const T* send_ptr = rank_ == root ? send.data() : nullptr;
		mpi_check(
			MPI_Scatterv(send_ptr, send_counts.data(), displacements.data(), mpi_type_v<T>,
						 recv.data(), recv.size(), mpi_type_v<T>, root, comm_),
			"MPI_Scatterv");
	}

	template <typename T>
	void gatherv(const std::vector<T>& send,
				 const std::vector<int>& recv_counts,
				 const std::vector<int>& displacements,
				 std::vector<T>& recv,
				 int root = 0) const {
		T* recv_ptr = rank_ == root ? recv.data() : nullptr;
		mpi_check(
			MPI_Gatherv(send.data(), send.size(), mpi_type_v<T>, recv_ptr,
						recv_counts.data(), displacements.data(), mpi_type_v<T>, root, comm_),
			"MPI_Gatherv");
	}

	double reduce_max(double lval, int root = 0) const {
		double gval = 0.0;
		mpi_check(
			MPI_Reduce(&lval, &gval, 1, MPI_DOUBLE, MPI_MAX, root, comm_),
			"MPI_Reduce(MAX)");
		return gval;
	}

private:
	MPI_Comm comm_;
	int rank_ = 0;
	int size_ = 1;
};

class WallTimer final {
public:
	WallTimer()
		: start_(MPI_Wtime()) {
	}

	void reset() {
		start_ = MPI_Wtime();
	}

	double elapsed_seconds() const {
		return MPI_Wtime() - start_;
	}

private:
	double start_ = 0.0;
};

struct TimingItem {
	std::string stage;
	double seconds = 0.0;
};

class DistributedTimer final {
public:
	explicit DistributedTimer(const MPIWorld& world)
		: world_(world) {
	}

	template <typename Func>
	void measure(const std::string& stage, Func&& task) {
		world_.barrier();
		WallTimer timer;
		std::forward<Func>(task)();
		world_.barrier();
		const double stage_seconds = world_.reduce_max(timer.elapsed_seconds());
		if (world_.rank() == 0) {
			records_.emplace_back(TimingItem{ stage, stage_seconds });
		}
	}

	const std::vector<TimingItem>& records() const noexcept {
		return records_;
	}

private:
	const MPIWorld& world_;
	std::vector<TimingItem> records_;
};
