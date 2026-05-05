#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <pthread.h>

struct PipelineCtx final {
	double a = 0.0;
	double b = 0.0;
	double c = 0.0;

	double b_sq = 0.0;
	double ac_4 = 0.0;

	double D = 0.0;
	double sqrt_D = 0.0;
	double x1 = 0.0;
	double x2 = 0.0;

	pthread_mutex_t mutex{};
	pthread_cond_t cond_bsq{};
	pthread_cond_t cond_4ac{};
	pthread_cond_t cond_D{};
	pthread_cond_t cond_sqrt{};
	pthread_cond_t cond_done{};

	bool bsq_ready = false;
	bool ac4_ready = false;
	bool D_ready = false;
	bool sqrt_ready = false;
	bool x1_ready = false;
	bool x2_ready = false;

	std::exception_ptr error;
};

void* calc_bsq(void* arg) {
	auto& ctx = *static_cast<PipelineCtx*>(arg);
	try {
		pthread_mutex_lock(&ctx.mutex);
		ctx.b_sq = ctx.b * ctx.b;
		ctx.bsq_ready = true;
		pthread_cond_signal(&ctx.cond_bsq);
		pthread_mutex_unlock(&ctx.mutex);
	}
	catch (...) {
		ctx.error = std::current_exception();
		ctx.bsq_ready = true;
		pthread_cond_signal(&ctx.cond_bsq);
		pthread_mutex_unlock(&ctx.mutex);
	}
	return nullptr;
}

void* calc_4ac(void* arg) {
	auto& ctx = *static_cast<PipelineCtx*>(arg);
	try {
		pthread_mutex_lock(&ctx.mutex);
		ctx.ac_4 = 4.0 * ctx.a * ctx.c;
		ctx.ac4_ready = true;
		pthread_cond_signal(&ctx.cond_4ac);
		pthread_mutex_unlock(&ctx.mutex);
	}
	catch (...) {
		ctx.error = std::current_exception();
		ctx.ac4_ready = true;
		pthread_cond_signal(&ctx.cond_4ac);
		pthread_mutex_unlock(&ctx.mutex);
	}
	return nullptr;
}

void* calc_D(void* arg) {
	auto& ctx = *static_cast<PipelineCtx*>(arg);
	try {
		pthread_mutex_lock(&ctx.mutex);

		while (!ctx.bsq_ready) {
			pthread_cond_wait(&ctx.cond_bsq, &ctx.mutex);
		}
		if (ctx.error) {
			ctx.D_ready = true;
			pthread_cond_signal(&ctx.cond_D);
			pthread_mutex_unlock(&ctx.mutex);
			return nullptr;
		}

		while (!ctx.ac4_ready) {
			pthread_cond_wait(&ctx.cond_4ac, &ctx.mutex);
		}
		if (ctx.error) {
			ctx.D_ready = true;
			pthread_cond_signal(&ctx.cond_D);
			pthread_mutex_unlock(&ctx.mutex);
			return nullptr;
		}

		ctx.D = ctx.b_sq - ctx.ac_4;
		ctx.D_ready = true;
		pthread_cond_signal(&ctx.cond_D);
		pthread_mutex_unlock(&ctx.mutex);
	}
	catch (...) {
		ctx.error = std::current_exception();
		ctx.D_ready = true;
		pthread_cond_signal(&ctx.cond_D);
		pthread_mutex_unlock(&ctx.mutex);
	}
	return nullptr;
}

void* calc_sqrtD(void* arg) {
	auto& ctx = *static_cast<PipelineCtx*>(arg);
	try {
		pthread_mutex_lock(&ctx.mutex);

		while (!ctx.D_ready) {
			pthread_cond_wait(&ctx.cond_D, &ctx.mutex);
		}
		if (ctx.error) {
			ctx.sqrt_ready = true;
			pthread_cond_broadcast(&ctx.cond_sqrt);
			pthread_mutex_unlock(&ctx.mutex);
			return nullptr;
		}

		ctx.sqrt_D = std::sqrt(ctx.D);
		ctx.sqrt_ready = true;
		pthread_cond_broadcast(&ctx.cond_sqrt);
		pthread_mutex_unlock(&ctx.mutex);
	}
	catch (...) {
		ctx.error = std::current_exception();
		ctx.sqrt_ready = true;
		pthread_cond_broadcast(&ctx.cond_sqrt);
		pthread_mutex_unlock(&ctx.mutex);
	}
	return nullptr;
}

void* calc_x1(void* arg) {
	auto& ctx = *static_cast<PipelineCtx*>(arg);
	try {
		pthread_mutex_lock(&ctx.mutex);

		while (!ctx.sqrt_ready) {
			pthread_cond_wait(&ctx.cond_sqrt, &ctx.mutex);
		}
		if (ctx.error) {
			ctx.x1_ready = true;
			pthread_cond_broadcast(&ctx.cond_done);
			pthread_mutex_unlock(&ctx.mutex);
			return nullptr;
		}

		ctx.x1 = (-ctx.b + ctx.sqrt_D) / (2.0 * ctx.a);
		ctx.x1_ready = true;
		pthread_cond_broadcast(&ctx.cond_done);
		pthread_mutex_unlock(&ctx.mutex);
	}
	catch (...) {
		ctx.error = std::current_exception();
		ctx.x1_ready = true;
		pthread_cond_broadcast(&ctx.cond_done);
		pthread_mutex_unlock(&ctx.mutex);
	}
	return nullptr;
}

void* calc_x2(void* arg) {
	auto& ctx = *static_cast<PipelineCtx*>(arg);
	try {
		pthread_mutex_lock(&ctx.mutex);

		while (!ctx.sqrt_ready) {
			pthread_cond_wait(&ctx.cond_sqrt, &ctx.mutex);
		}
		if (ctx.error) {
			ctx.x2_ready = true;
			pthread_cond_broadcast(&ctx.cond_done);
			pthread_mutex_unlock(&ctx.mutex);
			return nullptr;
		}

		ctx.x2 = (-ctx.b - ctx.sqrt_D) / (2.0 * ctx.a);
		ctx.x2_ready = true;
		pthread_cond_broadcast(&ctx.cond_done);
		pthread_mutex_unlock(&ctx.mutex);
	}
	catch (...) {
		ctx.error = std::current_exception();
		ctx.x2_ready = true;
		pthread_cond_broadcast(&ctx.cond_done);
		pthread_mutex_unlock(&ctx.mutex);
	}
	return nullptr;
}

void pthread_check(int code, std::string_view operation) {
	if (code != 0) {
		throw std::runtime_error(std::string(operation) + " failed.");
	}
}

int main() {
	double a = 0.0;
	double b = 0.0;
	double c = 0.0;
	std::cin >> a >> b >> c;

	// baseline
	auto start = std::chrono::steady_clock::now();
	const double D = b * b - 4.0 * a * c;
	const double sqrtD = std::sqrt(D);
	const double x1 = (-b + sqrtD) / (2.0 * a);
	const double x2 = (-b - sqrtD) / (2.0 * a);
	auto end = std::chrono::steady_clock::now();
	const double baseline_time = std::chrono::duration<double>(end - start).count();

	// pipeline
	PipelineCtx ctx;
	ctx.a = a;
	ctx.b = b;
	ctx.c = c;

	pthread_mutex_init(&ctx.mutex, nullptr);
	pthread_cond_init(&ctx.cond_bsq, nullptr);
	pthread_cond_init(&ctx.cond_4ac, nullptr);
	pthread_cond_init(&ctx.cond_D, nullptr);
	pthread_cond_init(&ctx.cond_sqrt, nullptr);
	pthread_cond_init(&ctx.cond_done, nullptr);

	pthread_t t_bsq, t_4ac, t_D, t_sqrt, t_x1, t_x2;

	start = std::chrono::steady_clock::now();

	pthread_check(pthread_create(&t_bsq, nullptr, calc_bsq, &ctx), "pthread_create(bsq)");
	pthread_check(pthread_create(&t_4ac, nullptr, calc_4ac, &ctx), "pthread_create(4ac)");

	pthread_check(pthread_create(&t_D, nullptr, calc_D, &ctx), "pthread_create(D)");
	pthread_check(pthread_create(&t_sqrt, nullptr, calc_sqrtD, &ctx), "pthread_create(sqrt)");
	pthread_check(pthread_create(&t_x1, nullptr, calc_x1, &ctx), "pthread_create(x1)");
	pthread_check(pthread_create(&t_x2, nullptr, calc_x2, &ctx), "pthread_create(x2)");

	pthread_check(pthread_join(t_bsq, nullptr), "pthread_join(bsq)");
	pthread_check(pthread_join(t_4ac, nullptr), "pthread_join(4ac)");
	pthread_check(pthread_join(t_D, nullptr), "pthread_join(D)");
	pthread_check(pthread_join(t_sqrt, nullptr), "pthread_join(sqrt)");
	pthread_check(pthread_join(t_x1, nullptr), "pthread_join(x1)");
	pthread_check(pthread_join(t_x2, nullptr), "pthread_join(x2)");

	end = std::chrono::steady_clock::now();

	if (ctx.error) std::rethrow_exception(ctx.error);

	const double pipeline_time = std::chrono::duration<double>(end - start).count();

	pthread_cond_destroy(&ctx.cond_done);
	pthread_cond_destroy(&ctx.cond_sqrt);
	pthread_cond_destroy(&ctx.cond_D);
	pthread_cond_destroy(&ctx.cond_4ac);
	pthread_cond_destroy(&ctx.cond_bsq);
	pthread_mutex_destroy(&ctx.mutex);

	std::cout << std::fixed << std::setprecision(6);
	std::cout << x1 << ' ' << x2 << ' ' << baseline_time << '\n';
	std::cout << ctx.x1 << ' ' << ctx.x2 << ' ' << pipeline_time << std::endl;

	return 0;
}
