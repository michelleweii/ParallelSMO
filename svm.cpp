#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <float.h>
#include <string.h>
#include <stdarg.h>
#include <limits.h>
#include <locale.h>
#include<Windows.h>
#include<iostream>
#include<iomanip>
#include<fstream>
#include<time.h>
#include "svm.h"
using std::cout;
using std::endl;

int libsvm_version = LIBSVM_VERSION;
typedef float Qfloat;
typedef signed char schar;
#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif
#ifndef min3 //比较3个数中的最小
template <class T> static inline T min3 (T x,T y,T z)
{ 
	T d=min(x,y);
	return min(d,z);
}
#endif
#ifndef max3
template <class T> static inline T max3 (T x,T y,T z) 
{ 
	T d=max(x,y);
	return max(d,z);
}
#endif
template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
template <class S, class T> static inline void clone(T*& dst, S* src, int n)
{
	dst = new T[n];
	memcpy((void *)dst,(void *)src,sizeof(T)*n);
}
static inline double powi(double base, int times)
{
	double tmp = base, ret = 1.0;

	for(int t=times; t>0; t/=2)
	{
		if(t%2==1) ret*=tmp;
		tmp = tmp * tmp;
	}
	return ret;
}
#define INF HUGE_VAL
#define TAU 1e-12
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
double multi_classTime = 0; 
double total_solver_time=0;// include all kinds of svm_train_one
int total_iter=0;// total # of iters.
char * output_result_file;
FILE * fout;
FILE*  fout_obj; //2014-10-30：保存每次迭代后的obj值
FILE*  fout_flag;//保存每次迭代采用的方式：parallel or single
//cache的统计信息(只算working set selection阶段的)
long totalVisits;
long totalCacheHits;
long totalCacheElements;

/*测试用*/
int shrink_times = 0;




static void print_string_stdout(const char *s)
{
	fputs(s,stdout);
	fflush(stdout);
}
static void (*svm_print_string) (const char *) = &print_string_stdout;
#if 1
static void info(const char *fmt,...)
{
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*svm_print_string)(buf);
}
#else
static void info(const char *fmt,...) {}
#endif

//
// Kernel Cache
//
// l is the number of total data items
// size is the cache size limit in bytes
//
class Cache
{
public:
	Cache(int l,long int size);
	~Cache();

	// request data [0,len)
	// return some position p where [p,len) need to be filled
	// (p >= len if nothing needs to be filled)
	/*20171219-索引idx对应的get_Q是否已缓存*/
	bool is_cached(int idx, int len); 
	Qfloat Cache::get_Q_ij(int i, int j); /*若有缓存,返回已有的值, 否则返回NULL*/
	int get_data(const int index, Qfloat **data, int len);
	int get_data(const int index, Qfloat **data, int len,int flag);//加个参数的，只当flag为1，统计命中情况
	void swap_index(int i, int j);
	void getHitInfo(long&totCnts,long& hitCnts,long& hitEls)
    {
		totCnts=totalCounts;
		hitCnts=hit.hitCounts;
		hitEls=hit.hitElements;
	}
private:
	int l;
	long int size;
	struct head_t
	{
		head_t *prev, *next;	// a circular list
		Qfloat *data;
		int len;		// data[0,len) is cached in this entry
	};
	struct Hit  //统计cache的命中次数和命中Qij的元素个数
	{   
		long hitCounts;//仅统计工作集选择时的命中，且若i在选择工作集时命中，则在接下来的因更新工作集时的命中(肯定命中)则不算在内
		long hitElements;
	}hit;
	
	long totalCounts;//总的访问cache次数：注意，仅统计工作集选择时的访问
	head_t *head;
	head_t lru_head;
	void lru_delete(head_t *h);
	void lru_insert(head_t *h);
};

bool Cache::is_cached(int idx, int len) {
	head_t *h = &head[idx];
	if (h->len >= len)
		return true;
	else
		return false;
}

Cache::Cache(int l_,long int size_):l(l_)
{   
	size = size_;
	totalCounts = 0;
	hit.hitCounts=0; //++
	hit.hitElements=0;//++
	head = (head_t *)calloc(l,sizeof(head_t));	// initialized to 0
	size /= sizeof(Qfloat);
	size -= l * sizeof(head_t) / sizeof(Qfloat);
	size = max(size, 2 * (long int) l);	// cache must be large enough for two columns
	lru_head.next = lru_head.prev = &lru_head;
}

Cache::~Cache()
{
	for(head_t *h = lru_head.next; h != &lru_head; h=h->next)
		free(h->data);
	free(head);
}

void Cache::lru_delete(head_t *h)
{
	// delete from current location
	h->prev->next = h->next;
	h->next->prev = h->prev;
}

void Cache::lru_insert(head_t *h)
{
	// insert to last position
	h->next = &lru_head;
	h->prev = lru_head.prev;
	h->prev->next = h;
	h->next->prev = h;
}

Qfloat Cache::get_Q_ij(int i, int j) {
	head_t *h = &head[i];
	if (h->len && h->len >= j)
		return h->data[j];
	else
		return NULL;
}

int Cache::get_data(const int index, Qfloat **data, int len)
{
	head_t *h = &head[index];
	if(h->len) 
	{
		lru_delete(h);
	}
	int more = len - h->len;

	if(more > 0)
	{
		// free old space
		while(size < more)
		{
			head_t *old = lru_head.next;
			lru_delete(old);
			free(old->data);
			size += old->len;
			old->data = 0;
			old->len = 0;
		}

		// allocate new space
		h->data = (Qfloat *)realloc(h->data,sizeof(Qfloat)*len);
		size -= more;
		swap(h->len,len);
	}

	lru_insert(h);
	*data = h->data;
	return len;
}
int Cache::get_data(const int index, Qfloat **data, int len,int flag)
{
	head_t *h = &head[index];
	if(h->len) //++
	{
		lru_delete(h);
		if(flag)//
		{
		  hit.hitCounts++;
		  hit.hitElements+= (h->len) < len? (h->len):len;  
		}
	}
	if(flag) totalCounts++;//在选择工作集时总的访问次数，应等于iters*2.
	int more = len - h->len;
	if(more > 0)
	{
		// free old space
		while(size < more)
		{
			head_t *old = lru_head.next;
			lru_delete(old);
			free(old->data);
			size += old->len;
			old->data = 0;
			old->len = 0;
		}

		// allocate new space
		h->data = (Qfloat *)realloc(h->data,sizeof(Qfloat)*len);
		size -= more;
		swap(h->len,len);
	}

	lru_insert(h);
	*data = h->data;
	return len;
}
void Cache::swap_index(int i, int j)
{
	if(i==j) return;

	if(head[i].len) lru_delete(&head[i]);
	if(head[j].len) lru_delete(&head[j]);
	swap(head[i].data,head[j].data);
	swap(head[i].len,head[j].len);
	if(head[i].len) lru_insert(&head[i]);
	if(head[j].len) lru_insert(&head[j]);

	if(i>j) swap(i,j);
	for(head_t *h = lru_head.next; h!=&lru_head; h=h->next)
	{
		if(h->len > i)
		{
			if(h->len > j)
				swap(h->data[i],h->data[j]);
			else
			{
				// give up
				lru_delete(h);
				free(h->data);
				size += h->len;
				h->data = 0;
				h->len = 0;
			}
		}
	}
}

//
// Kernel evaluation
//
// the static method k_function is for doing single kernel evaluation
// the constructor of Kernel prepares to calculate the l*l kernel matrix
// the member function get_Q is for getting one column from the Q Matrix
//
class QMatrix {
public:
	virtual Qfloat *get_Q(int column, int len,int flag) const { return 0;};  //只为了能统计cache命中情况
	virtual void getHitInfo(long&totCnts,long& hitCnts,long& hitEls) const{}
	virtual Qfloat *get_Q(int column, int len) const = 0;
	virtual double *get_QD() const = 0;
	virtual void swap_index(int i, int j) const = 0;
	virtual Qfloat get_Q_ij(int i, int j) const { return 0; }; /*求单个Qij的值*/
	virtual bool is_cached(int i, int len) const { return false; }; /*样本i对应的Q_i是否已缓存*/
	virtual ~QMatrix() {}
};

class Kernel: public QMatrix {
public:
	Kernel(int l, svm_node * const * x, const svm_parameter& param);
	virtual ~Kernel();

	static double k_function(const svm_node *x, const svm_node *y,
				 const svm_parameter& param);
	virtual Qfloat *get_Q(int column, int len) const = 0;
	virtual double *get_QD() const = 0;
	virtual void swap_index(int i, int j) const	// no so const...
	{
		swap(x[i],x[j]);
		if(x_square) swap(x_square[i],x_square[j]);
	}
protected:

	double (Kernel::*kernel_function)(int i, int j) const;

private:
	const svm_node **x;
	double *x_square;

	// svm_parameter
	const int kernel_type;
	const int degree;
	const double gamma;
	const double coef0;

	static double dot(const svm_node *px, const svm_node *py);
	double kernel_linear(int i, int j) const
	{
		return dot(x[i],x[j]);
	}
	double kernel_poly(int i, int j) const
	{
		return powi(gamma*dot(x[i],x[j])+coef0,degree);
	}
	double kernel_rbf(int i, int j) const
	{
		return exp(-gamma*(x_square[i]+x_square[j]-2*dot(x[i],x[j])));
	}
	double kernel_sigmoid(int i, int j) const
	{
		return tanh(gamma*dot(x[i],x[j])+coef0);
	}
	double kernel_precomputed(int i, int j) const
	{
		return x[i][(int)(x[j][0].value)].value;
	}
};

Kernel::Kernel(int l, svm_node * const * x_, const svm_parameter& param)
:kernel_type(param.kernel_type), degree(param.degree),
 gamma(param.gamma), coef0(param.coef0)
{
	switch(kernel_type)
	{
		case LINEAR:
			kernel_function = &Kernel::kernel_linear;
			break;
		case POLY:
			kernel_function = &Kernel::kernel_poly;
			break;
		case RBF:
			kernel_function = &Kernel::kernel_rbf;
			break;
		case SIGMOID:
			kernel_function = &Kernel::kernel_sigmoid;
			break;
		case PRECOMPUTED:
			kernel_function = &Kernel::kernel_precomputed;
			break;
	}

	clone(x,x_,l);

	if(kernel_type == RBF)
	{
		x_square = new double[l];
		for(int i=0;i<l;i++)
			x_square[i] = dot(x[i],x[i]);
	}
	else
		x_square = 0;
}

Kernel::~Kernel()
{
	delete[] x;
	delete[] x_square;
}

double Kernel::dot(const svm_node *px, const svm_node *py)
{
	double sum = 0;
	while(px->index != -1 && py->index != -1)
	{
		if(px->index == py->index)
		{
			sum += px->value * py->value;
			++px;
			++py;
		}
		else
		{
			if(px->index > py->index)
				++py;
			else
				++px;
		}			
	}
	return sum;
}

double Kernel::k_function(const svm_node *x, const svm_node *y,
			  const svm_parameter& param)
{
	switch(param.kernel_type)
	{
		case LINEAR:
			return dot(x,y);
		case POLY:
			return powi(param.gamma*dot(x,y)+param.coef0,param.degree);
		case RBF:
		{
			double sum = 0;
			while(x->index != -1 && y->index !=-1)
			{
				if(x->index == y->index)
				{
					double d = x->value - y->value;
					sum += d*d;
					++x;
					++y;
				}
				else
				{
					if(x->index > y->index)
					{	
						sum += y->value * y->value;
						++y;
					}
					else
					{
						sum += x->value * x->value;
						++x;
					}
				}
			}

			while(x->index != -1)
			{
				sum += x->value * x->value;
				++x;
			}

			while(y->index != -1)
			{
				sum += y->value * y->value;
				++y;
			}
			
			return exp(-param.gamma*sum);
		}
		case SIGMOID:
			return tanh(param.gamma*dot(x,y)+param.coef0);
		case PRECOMPUTED:  //x: test (validation), y: SV
			return x[(int)(y->value)].value;
		default:
			return 0;  // Unreachable 
	}
}

// An SMO algorithm in Fan et al., JMLR 6(2005), p. 1889--1918
// Solves:
//
//	min 0.5(\alpha^T Q \alpha) + p^T \alpha
//
//		y^T \alpha = \delta
//		y_i = +1 or -1
//		0 <= alpha_i <= Cp for y_i = 1
//		0 <= alpha_i <= Cn for y_i = -1
//
// Given:
//
//	Q, p, y, Cp, Cn, and an initial feasible point \alpha
//	l is the size of vectors and matrices
//	eps is the stopping tolerance
//
// solution will be put in \alpha, objective value will be put in obj
//
class Solver {
public:
	Solver() {};
	virtual ~Solver() {};

	struct SolutionInfo {
		double obj;
		double rho;
		double upper_bound_p;
		double upper_bound_n;
		double r;	// for Solver_NU
	};
	/*单路:non-shrking+non-cache*/
	void Solve2(int l, const QMatrix& Q, const double *p_, const schar *y_,
		   double *alpha_, double Cp, double Cn, double eps,
		   SolutionInfo* si);
	/*单路: shrinking+cache*/
	void Solve(int l, const QMatrix& Q, const double *p_, const schar *y_,
		   double *alpha_, double Cp, double Cn, double eps,
		   SolutionInfo* si, int shrinking);
	/*多路并行*/
	void Solve3(int l, const QMatrix& Q, const double *p_, const schar *y_,
		double *alpha_, double Cp, double Cn, double eps,
		SolutionInfo* si);
protected:
	int active_size;
	schar *y;
	double *G;		// gradient of objective function
	enum { LOWER_BOUND, UPPER_BOUND, FREE };
	char *alpha_status;	// LOWER_BOUND, UPPER_BOUND, FREE
	double *alpha;
	const QMatrix *Q;
	const double *QD;
	double eps;
	double Cp,Cn;
	double *p;
	int *active_set;
	double *G_bar;		// gradient, if we treat free variables as 0
	double* *iG_bar; // 用于在并行中累加更新的iG_bar值.
	int l; //样本集大小
	bool unshrink;	// XXX
	double old_obj;//存放上一次迭代计算出的目标函数值.
	int len; 
	int iter; //++
	double* *iG;//用于累加并行计算时更新的G值，每轮更新后需记得清0
	double* deltaG_Iup; //ncpu*1, 用于筛选多路违反对时, Iup索引对应的iG
	double* *deltaG; //ncpu*ncpu, 用于筛选多路违反对时，局部更新imin中索引对应的iG
	double  *deltaAlpha; //存放并行时, 更新的alpha量
	int ncpu; //cpu数
	int * iup;// 存放并行时,每一迭代中，前N个Iup 索引
	int * IUP; //存放并行时，每一次迭代中，实际筛选得到的iup索引
	int * ILOW;//存放并行时，每一次迭代中，实际筛选得到的ilow索引
	int** imin; /*iup对应的候选ilow数组*/
	double* *obj_diff_min;//每个Iup对应的候选Ilow的obj_diff，有序
	int* vflag;//flag[i]=1: 在样本在iter中已被访问到; flag[i]=0: 未被访问(length=l)
	bool* iG_bar_update_flag; //在并行中标记各路数的iG_bar是否有效
	bool* old_is_upper_bound; //当使用shrink时，在并行版本中，需要暂时保存违反因子更新前是否为'is_upper_bound',用于后续更新G_bar
	double* nFiup;//存放前ncpu个最大 -Fi(i属于Iup):从大到小

	double get_C(int i)
	{
		return (y[i] > 0)? Cp : Cn;
	}
	void update_alpha_status(int i)
	{
		if(alpha[i] >= get_C(i))
			alpha_status[i] = UPPER_BOUND;
		else if(alpha[i] <= 0)
			alpha_status[i] = LOWER_BOUND;
		else alpha_status[i] = FREE;
	}
	bool is_upper_bound(int i) { return alpha_status[i] == UPPER_BOUND; }
	bool is_lower_bound(int i) { return alpha_status[i] == LOWER_BOUND; }
	bool is_free(int i) { return alpha_status[i] == FREE; }
	void swap_index(int i, int j);
	void reconstruct_gradient();
	virtual int libsvm_select_working_set(int &i, int &j);
	virtual int wss35_select_working_set(int &out_i, int &out_j);
	/*不是新的工作集选择算法，只是在函数内部直接使用libsvm或wss35， 在多路并行更新中使用*/
	virtual int select_working_set3(int &out_i, int &out_j, double& total_sel_pair_time);
	virtual double calculate_rho();
	virtual void do_shrinking();
private:
	bool be_shrunk(int i, double Gmax1, double Gmax2);
	int select_kmax_Iup(int parallels);
	void select_k_ilow_candidates(int parallel_idx, int cached_flag);
	int select_k_working_set(double& total_sel_pair_time); 
	void solve3_init_func(int l, const QMatrix& Q, const double *p_, const schar *y_,
		double *alpha_, double Cp, double Cn, double eps, double& multi_classTime);
	void solve3_free_func(int max_iter, double* alpha_, SolutionInfo* si,double& multi_classTime);
	void update_alpha(int m, int n, double deltaGi, double deltaGj);
	void iipsmo_update_G(int nprls, double& total_updateG_time);
	void ipsmo_update_G(int nprls, double& total_updateG_time);
};


int Solver::select_k_working_set(double& total_sel_pair_time) {

	LARGE_INTEGER litmp;
	QueryPerformanceFrequency(&litmp);
	double dfFreq = (double)litmp.QuadPart;// 获得计数器的时钟频率 

	QueryPerformanceCounter(&litmp);
	LONGLONG sel_kmax_Iup_start = litmp.QuadPart;

	len = select_kmax_Iup(this->ncpu); //先选出前Iup中前ncpu个最大的-Fi.(返回符合条件的Iup)
	
	/*迭代优化*/
	int cached_flag = 0;
	if (useShrinkAndCache && useCacheThrehold) {
		int cachedCount = 0;
		for (int t = 0; t < len; t++) {
			if ((this->Q)->is_cached(iup[t], active_size)) /*统计cached的路数*/
				cachedCount++;
		}
		if (cachedCount*1.0 >= len*parallel_cache_threhold)
			cached_flag = 1;
	}

	QueryPerformanceCounter(&litmp);
	LONGLONG  sel_kmax_Iup_end = litmp.QuadPart;
	double sel_kmax_Iup_time = (double)(sel_kmax_Iup_end - sel_kmax_Iup_start) / dfFreq;
	total_sel_pair_time += sel_kmax_Iup_time;

	int nprls = 0;
	double max_sel_k_Ilow_time = 0;

	for (int t = 0; t<len; t++)
	{
		double temp_sel_time = 0;
		deltaG_Iup[t] = 0;

		QueryPerformanceCounter(&litmp);
		LONGLONG select_k_ilow_start = litmp.QuadPart;

		/*对select_kmax_Iup()中选出的每一个Iup, 按照libsvm或wss35算法找出前ilowCandidateSize个候选Ilow*/
		select_k_ilow_candidates(t, cached_flag); 

		QueryPerformanceCounter(&litmp);
		LONGLONG select_k_ilow_end = litmp.QuadPart;
		temp_sel_time = (double)(select_k_ilow_end - select_k_ilow_start) / dfFreq;
		max_sel_k_Ilow_time = max(max_sel_k_Ilow_time, temp_sel_time); /*模拟并行*/
	}
	total_sel_pair_time += max_sel_k_Ilow_time;

	double cycle_sel_time = 0.0;
	if (parallel_algorithm == IIPSMO) { /*IIPSMO*/
		/*
		1.按照libsvm或wss35从剩余候选路数中选出一个最大违反对(i,j)
		2.计算(delta_i, delta_j)，并更新i和j的状态(串行)
		3.计算(delta_i, delta_j)对剩余候选路数中的因子的梯度影响，并重新更新违反状态(并行)
		4.重复上述步骤，直到找不出新的违反对
		5.更新多路违反因子对全局的梯度变化(并行，计算时各保留一份副本)
		6.汇总步骤5中的影响变化值
		*/
		for (int k1 = 0; k1 < len; k1++) {

			/*串行部分*/
			QueryPerformanceCounter(&litmp);
			LONGLONG sel_pair_start = litmp.QuadPart;
			int sel_idx = -1;
			double max_obj_min = -INF;
			for (int k2 = 0; k2 < len; k2++) {
				int i = iup[k2];
				/*
				 i == -1: 因子i已被选中
				 vflag[i] == 1: i在其它路中作为ilow被选中
				 imin[k2][0]、obj_diff_min[k2][0] == -INF: 该路当前找不出违反对
				*/
				if ((i == -1) || vflag[i] == 1 || imin[k2][0] < 0 || obj_diff_min[k2][0] == -INF)
					continue;

				if (obj_diff_min[k2][0] > max_obj_min) {
					sel_idx = k2;
					max_obj_min = obj_diff_min[k2][0];
				}
			}//end-for
			if (sel_idx == -1) /*找不出违反对*/
				break;
			int i = iup[sel_idx];
			int j = imin[sel_idx][0];

			IUP[nprls] = i;
			ILOW[nprls] = j;
			nprls++;

			iup[sel_idx] = -1;
			imin[sel_idx][0] = -1;

			vflag[i] = 1;
			vflag[j] = 1;
			/*求(delta_i, delta_j)*/
			this->update_alpha(i, j, deltaG_Iup[sel_idx], deltaG[sel_idx][0]);

			QueryPerformanceCounter(&litmp);
			LONGLONG sel_pair_end = litmp.QuadPart;
			double sel_pair_time = (double)(sel_pair_end - sel_pair_start) / dfFreq;

			double update_k_ilow_time = 0.0;

			/*局部更新剩余候选路数的G*/
			for (int k2 = 0; k2 < len; k2++) {

				QueryPerformanceCounter(&litmp);
				double update_k_ilow_start = litmp.QuadPart;

				int m = iup[k2];
				double temp_Gmax2;
				if ((m != -1) && vflag[m] != 1) {
					deltaG_Iup[k2] += deltaAlpha[i] * (this->Q->get_Q_ij(i, m)) + deltaAlpha[j] * (this->Q->get_Q_ij(j, m));

					double Gmax = -y[m] * (G[m] + deltaG_Iup[k2]); /*-Fm*/
					double Gmax2 = -INF;

					int Gmax_idx = -1; /*在更新过程中记录该路最大违反因子的索引*/
					double max_obj_diff = -INF;
					for (int k3 = 0; k3 < ilowCandidateSize; k3++) {

						obj_diff_min[k2][k3] = -INF;
						int n = imin[k2][k3];
						if (n == -1)
							continue;
						if (vflag[n] == 1) {
							imin[k2][k3] = -1;
							continue;
						}
						deltaG[k2][k3] += deltaAlpha[i] * (this->Q->get_Q_ij(i, n)) + deltaAlpha[j] * (this->Q->get_Q_ij(j, n));
						if (y[n] == +1) {
							if (!is_lower_bound(n)) {
								temp_Gmax2 = G[n] + deltaG[k2][k3];
								double grad_diff = Gmax + temp_Gmax2;
								if (temp_Gmax2 >= Gmax2){
									Gmax2 = temp_Gmax2;
								}
								if (grad_diff > my_eps)  //注: 这里使用了my_eps
								{
									double quad_coef = QD[m] + QD[n] - 2.0*y[m] * this->Q->get_Q_ij(m, n);
									if (quad_coef <= 0) quad_coef = TAU;

									if (useWSS35) {
										double temp_value = grad_diff / quad_coef;
										obj_diff_min[k2][k3] = temp_value * temp_value * grad_diff;
									}
									else
										obj_diff_min[k2][k3] = grad_diff * grad_diff / quad_coef;

									if (obj_diff_min[k2][k3] > max_obj_diff)
									{
										Gmax_idx = k3;
										max_obj_diff = obj_diff_min[k2][k3];
									}
								}
							}
						}
						else
						{
							if (!is_upper_bound(n)) {
								temp_Gmax2 = -(G[n] + deltaG[k2][k3]);
								double grad_diff = Gmax + temp_Gmax2;
								if (temp_Gmax2 >= Gmax2) {
									Gmax2 = temp_Gmax2;
								}
								if (grad_diff > my_eps)
								{
									double quad_coef = QD[m] + QD[n] + 2.0*y[m] * this->Q->get_Q_ij(m, n);
									if (quad_coef <= 0) quad_coef = TAU;

									if (useWSS35) {
										double temp_value = grad_diff / quad_coef;
										obj_diff_min[k2][k3] = temp_value * temp_value * grad_diff;
									}
									else
										obj_diff_min[k2][k3] = grad_diff * grad_diff / quad_coef;

									if (obj_diff_min[k2][k3] > max_obj_diff)
									{
										Gmax_idx = k3;
										max_obj_diff = obj_diff_min[k2][k3];
									}
								}
						
							}
						}//else
					}//end-for k3
					if (Gmax + Gmax2 >= eps) { /*该路该轮次找到了候选的违反对*/
						/*交换imin[k2][0]与imin[k2][Gmax_idx]的位置，
						  这样更新完从imin[k2][0]就能取到该路候选最大违反因子或不存在
						*/
						swap(obj_diff_min[k2][0], obj_diff_min[k2][Gmax_idx]);
						swap(imin[k2][0], imin[k2][Gmax_idx]);
						swap(deltaG[k2][0], deltaG[k2][Gmax_idx]);
					}
				}
				QueryPerformanceCounter(&litmp);
				double update_k_ilow_end = litmp.QuadPart;
				double temp_update_k_ilow_time = (double)(update_k_ilow_end - update_k_ilow_start) / dfFreq;
				update_k_ilow_time = max(update_k_ilow_time, temp_update_k_ilow_time);
			}/*end-for k2*/
			/*每轮筛选时间累加*/
			cycle_sel_time += sel_pair_time + update_k_ilow_time;
		}/*end-for k1*/
	}
	else { /*IPSMO/PSMO*/

		QueryPerformanceCounter(&litmp);
		LONGLONG sel_pair_start = litmp.QuadPart;

		for (int t = 0; t<len; t++)
		{  
			int i = iup[t];
			IUP[t] = i;
			ILOW[t] = -1;
			if ((i != -1) && vflag[i] != 1)
			{
				int j = 0;
				while ( j < ilowCandidateSize && imin[t][j] >= 0) //important!
				{
					int k = imin[t][j];
					if (vflag[k] != 1) {
						ILOW[t] = k;
						vflag[i] = 1;
						vflag[k] = 1; 
						break;
					}
					j++;
				}

			}
		}
		int idx = -1;
		for (int t = 0; t<len;){   
			if (IUP[t] != -1 && ILOW[t] != -1) {
				idx++;
				if (idx != t) {
					IUP[idx] = IUP[t];
					ILOW[idx] = ILOW[t];
				}
			}
			t++;
		}
		nprls = idx + 1;
		QueryPerformanceCounter(&litmp);
		LONGLONG sel_pair_end = litmp.QuadPart;
		cycle_sel_time = (double)(sel_pair_end - sel_pair_start) / dfFreq;
	}

	total_sel_pair_time += cycle_sel_time;

	if (printLog) {
		fprintf(fout, "iter=%d, nprls=%d, sel_kmax_Iup_time=%f, max_sel_k_Ilow_time=%f, cycle_sel_time=%f, avg_sel_pair_time=%f, total_sel_pair_time=%f\n", iter, nprls, sel_kmax_Iup_time, max_sel_k_Ilow_time, cycle_sel_time, (sel_kmax_Iup_time + max_sel_k_Ilow_time + cycle_sel_time) / nprls,total_sel_pair_time);
	}
	
	return nprls;
 
}
void Solver::solve3_init_func(int l, const QMatrix& Q, const double *p_, const schar *y_,
	double *alpha_, double Cp, double Cn, double eps, double& initial_time) {


	LARGE_INTEGER litmp;
	QueryPerformanceFrequency(&litmp);
	double dfFreq = (double)litmp.QuadPart;// 获得计数器的时钟频率 

	QueryPerformanceCounter(&litmp);
	LONGLONG initial_start = litmp.QuadPart;

	this->ncpu = parallelSize;
	this->iter = 0;
	//this->si = si;
	this->old_obj = 0;

	this->l = l;
	this->Q = &Q;
	this->QD = Q.get_QD();
	clone(this->p, p_, l);
	clone(this->y, y_, l);
	clone(this->alpha, alpha_, l);
	this->Cp = Cp;
	this->Cn = Cn;
	this->eps = eps;
	vflag = new int[l];
	deltaAlpha = new double[l]; //由于每个ialpha也只影响两个alpha,所以这里用一维数组即可
	alpha_status = new char[l];

	for (int i = 0; i< l; i++) {
		update_alpha_status(i);
		vflag[i] = 0; // 在每次iter后都应清0.
		deltaAlpha[i] = 0;
	}
	//初始化二维数组imin: ncpu*ncpu
	imin = new int*[ncpu];
	deltaG = new double*[ncpu];
	obj_diff_min = new double*[ncpu];
	deltaG_Iup = new double[ncpu];
	
	IUP = new int[ncpu];
	ILOW = new int[ncpu];
	for (int i = 0; i<ncpu; i++)
	{
		imin[i] = new int[ilowCandidateSize];
		deltaG[i] = new double[ilowCandidateSize];
		obj_diff_min[i] = new double[ilowCandidateSize];
		nFiup = new double[ncpu];
	}
	iG = new double*[ncpu];
	if (useShrinkAndCache) {
		iG_bar = new double*[ncpu];
		iG_bar_update_flag = new bool[ncpu];
	}

	iup = new int[ncpu];

	for (int i = 0; i<ncpu; i++)
	{
		iG[i] = new double[l];
		if (useShrinkAndCache) {
			iG_bar[i] = new double[l];
			iG_bar_update_flag[i] = false;
		}
		for (int j = 0; j<this->l; j++)
		{
			iG[i][j] = 0;
		}
		if (useShrinkAndCache) {
			for (int j = 0; j<this->l; j++)
				iG_bar[i][j] = 0;
		}
	}

	// initialize active set (for shrinking)
	{
		active_set = new int[l];
		for (int i = 0; i<l; i++)
			active_set[i] = i;
		active_size = l;
	}

	// initialize gradient
	{
		G = new double[l];
		
		if (useShrinkAndCache) {
			G_bar = new double[l];
			old_is_upper_bound = new bool[l];
		}
		int i;
		for (i = 0; i<l; i++)
		{
			G[i] = p[i];
			if(useShrinkAndCache) G_bar[i] = 0;
		}
		for (i = 0; i<l; i++) {
			if (!is_lower_bound(i))
			{

				const Qfloat *Q_i;
				if (useShrinkAndCache) 
					this->Q->get_Q(i, l, 0);
				else
					this->Q->get_Q(i, l);

				double alpha_i = alpha[i];
				int j;
				for (j = 0; j<l; j++)
					G[j] += alpha_i*Q_i[j];

				if (useShrinkAndCache) {
					if(is_upper_bound(i))
						for(j=0;j<l;j++)
						  G_bar[j] += get_C(i) * Q_i[j];
				}
				else {
					delete[] Q_i;
				}
				
			}
		}

	}
	QueryPerformanceCounter(&litmp);
	LONGLONG initial_end = litmp.QuadPart;
	initial_time = (double)(initial_end - initial_start) / dfFreq;
	//double initial_time = (double)(initial_end - initial_start) / dfFreq;
	//multiclass += initial_time;
		 
}
void Solver::solve3_free_func(int max_iter, double* alpha_, SolutionInfo* si, double& free_solve_time) {

	LARGE_INTEGER litmp;
	QueryPerformanceFrequency(&litmp);
	double dfFreq = (double)litmp.QuadPart;// 获得计数器的时钟频率 

	QueryPerformanceCounter(&litmp);
	LONGLONG exit_iter_start = litmp.QuadPart;
	if (iter >= max_iter)
	{
		if (active_size < l)
		{
			// reconstruct the whole gradient to calculate objective value
			reconstruct_gradient();
			active_size = l;
			info("*");
		}
		fprintf(stderr, "\nWARNING: reaching max number of iterations\n");
	}

	// calculate rho
	si->rho = calculate_rho();
	{
		double v = 0;
		int i;
		for (i = 0; i<l; i++)
			v += alpha[i] * (G[i] + p[i]);
		si->obj = v / 2;
	}

	// put back the solution
	{
		for (int i = 0; i<l; i++)
			alpha_[active_set[i]] = alpha[i];
	}

	si->upper_bound_p = Cp;
	si->upper_bound_n = Cn;

	delete[] deltaAlpha;
	delete[] p;
	delete[] y;
	delete[] alpha;
	delete[] alpha_status;
	delete[] active_set;
	delete[] G;
	delete[] deltaG_Iup;
	for (int i = 0; i<ncpu; i++)
	{
		delete[] iG[i];
		delete[] obj_diff_min[i];
		delete[] imin[i];
		delete[] deltaG[i];
		if(useShrinkAndCache) delete[] iG_bar[i];
	}
	delete[] iG;
	delete[] imin;
	delete[] obj_diff_min;
	delete[] deltaG;

	if (useShrinkAndCache) {
		delete[] G_bar;
		delete[] old_is_upper_bound;
		delete[] iG_bar;
	}
	QueryPerformanceCounter(&litmp);
	LONGLONG exit_iter_end = litmp.QuadPart;
	free_solve_time = (double)(exit_iter_end - exit_iter_start) / dfFreq;
	//double free_solve_time = (double)(exit_iter_end - exit_iter_start) / dfFreq;
	//multiclass += free_solve_time;

}
/*注: 只支持非cache版本*/
void Solver::ipsmo_update_G(int nprls, double& total_updateG_time) {

	if (useShrinkAndCache) {
		info("ERROR: IPSMO/PSMO: 请设置‘useShrinkAndCache = 0’再重新运行!\n");
		return;
	}
	LARGE_INTEGER litmp;
	QueryPerformanceFrequency(&litmp);
	double dfFreq = (double)litmp.QuadPart;// 获得计数器的时钟频率

	double sel_time1 = 0.0;
	double sel_time2 = 0.0;
	double sel_time3 = 0.0;
	for (int t = 0; t < nprls; t++)
	{

		QueryPerformanceCounter(&litmp);
		LONGLONG iter_start1 = litmp.QuadPart;
		{
			int i = IUP[t];
			int j = ILOW[t];
			this->update_alpha(i, j, 0.0, 0.0);
	
			const Qfloat* Q_i = this->Q->get_Q(i, active_size);
			const Qfloat* Q_j = this->Q->get_Q(j, active_size);

			for (int k = 0; k < active_size; k++)
			{
				iG[t][k] = Q_i[k] * deltaAlpha[i] + Q_j[k] * deltaAlpha[j];
			}
		
			delete[] Q_i;
			delete[] Q_j;
			
		}
		QueryPerformanceCounter(&litmp);
		LONGLONG iter_end1 = litmp.QuadPart;
		double tmp_sel_time1 = (double)(iter_end1 - iter_start1) / dfFreq;
		sel_time1 = max(sel_time1, tmp_sel_time1);
	}
	if (nprls > 1)
	{
		if (parallel_algorithm == IPSMO)
		{    
			/*v1和v2可以并行求*/
			QueryPerformanceCounter(&litmp);
			LONGLONG iter_start2_1 = litmp.QuadPart;

			for (int t = 0; t < nprls - 1; t++)
				for (int z = 0; z < active_size; z++)
					iG[nprls - 1][z] += iG[t][z];

			double v1 = 0;
			/*attention: 在使用shrinking的前提下, active_size不一定等于l, 所以v1和v2并非真正的目标函数值*/
			for (int z = 0; z < active_size; z++)
				v1 += (alpha[z]) * (G[z] + iG[nprls - 1][z] + p[z]);
			
			QueryPerformanceCounter(&litmp);
			LONGLONG iter_end2_1 = litmp.QuadPart;
			double sel_time2_1 = (double)(iter_end2_1 - iter_start2_1) / dfFreq;

			QueryPerformanceCounter(&litmp);
			double iter_start2_2 = litmp.QuadPart;

			double v2 = 0.0;
			for (int z = 0; z < active_size; z++)
				v2 += (alpha[z]) * (G[z] + iG[0][z] + p[z]);

			for (int t = 1; t < nprls; t++) {
				int i = IUP[t];
				int j = ILOW[t];
				v2 = v2 - deltaAlpha[i] * (G[i] + iG[0][i] + p[i]) - deltaAlpha[j] * (G[j] + iG[0][j] + p[j]);
			}

			QueryPerformanceCounter(&litmp);
			LONGLONG iter_end2_2 = litmp.QuadPart;
			double sel_time2_2 = (double)(iter_end2_2 - iter_start2_2) / dfFreq;

			sel_time2 = max(sel_time2_1, sel_time2_2);

			if (v1 > v2) /*串行效果更好*/
			{
				QueryPerformanceCounter(&litmp);
				double sel_start2_3 = litmp.QuadPart;
				{
					/*还原除了1路以外的其它路的deltaAlpha*/
					for (int t = 1; t < nprls; t++) {
						int i = IUP[t];
						int j = ILOW[t];
						alpha[i] = alpha[i] - deltaAlpha[i];
						alpha[j] = alpha[j] - deltaAlpha[j];
						vflag[i] = 0;
						vflag[j] = 0;
					}
					nprls = 1; /*采用1路*/
				}
				QueryPerformanceCounter(&litmp);
				double sel_end2_3 = litmp.QuadPart;
				sel_time2 += (double)(sel_end2_3 - sel_start2_3) / dfFreq;
			 }
		}
		QueryPerformanceCounter(&litmp);
		double sel_start3 = litmp.QuadPart;
		{   
		
			for (int t = 0; t < nprls; t++) {
				int i = IUP[t];
				int j = ILOW[t];
				update_alpha_status(i);
				update_alpha_status(j);
				vflag[i] = 0;
				vflag[j] = 0;
				//fprintf(fout_obj, "(%d, %d)\n", i, j);
			}
			if (parallel_algorithm == IPSMO) {
				/*对IPSMO，在计算v1时已经将iG[]汇总到iG[nprls-1]中*/
				for (int z = 0; z < active_size; z++)
					G[z] += iG[nprls - 1][z];

			}else{
				for (int t = 0; t < nprls; t++)
					for (int z = 0; z < active_size; z++)
						G[z] += iG[t][z];
			}
		}
		QueryPerformanceCounter(&litmp);
		LONGLONG sel_end3 = litmp.QuadPart;
		sel_time3 = (double)(sel_end3 - sel_start3) / dfFreq;
	}
	else { /*nprls == 1*/

		QueryPerformanceCounter(&litmp);
		LONGLONG sel_start2 = litmp.QuadPart;

		update_alpha_status(IUP[0]);
		update_alpha_status(ILOW[0]);
		for (int z = 0; z < active_size; z++)
			G[z] = G[z] + iG[0][z];

		vflag[IUP[0]] = 0;
		vflag[ILOW[0]] = 0;

		//fprintf(fout_obj, "serial: (%d, %d)\n", IUP[0], ILOW[0]);

		QueryPerformanceCounter(&litmp);
		LONGLONG sel_end2 = litmp.QuadPart;
		sel_time2 = (double)(sel_end2 - sel_start2) / dfFreq;
	}

	total_updateG_time += sel_time1 + sel_time2 + sel_time3;

	if (printLog) {

		double v1 = 0;
		for (int z = 0; z<l; z++)
			v1 += alpha[z] * (G[z] + p[z]);
		double obj = v1 / 2;
		info("iter: %d,target_obj: %f\n", iter, obj);
		fprintf(fout_obj, "(%d, %f)\n", iter, obj);
		fprintf(fout_flag, "(%d, %d)\n", iter, nprls);
		fprintf(fout, ", update_G_time=%f, total_updateG_time=%f\n", (sel_time1+sel_time2+sel_time3), total_updateG_time);
	}

}


void Solver::iipsmo_update_G(int nprls, double& total_updateG_time) {

	LARGE_INTEGER litmp;
	QueryPerformanceFrequency(&litmp);
	double dfFreq = (double)litmp.QuadPart; 
	
	double updateG_time = 0.0;
	for (int t = 0; t<nprls; t++)
	{

		/*并行部分*/
		QueryPerformanceCounter(&litmp);
		LONGLONG updateG_start = litmp.QuadPart;

		int i = IUP[t];
		int j = ILOW[t];
		vflag[i] = 0; /*恢复访问标识*/
		vflag[j] = 0;

		//fprintf(fout_obj, "(%d, %d)\n", i, j);

		update_alpha_status(i);
		update_alpha_status(j);

		
		const Qfloat *Q_i = NULL;
		const Qfloat *Q_j = NULL;

		if (!useShrinkAndCache) {/*non-cache*/
			Q_i = this->Q->get_Q(i, active_size);
			Q_j = this->Q->get_Q(j, active_size);
		}else{/*cache*/
			Q_i = this->Q->get_Q(i, active_size, 1);
			Q_j = this->Q->get_Q(j, active_size, 1);
		}

		if (nprls > 1) {
			for (int k = 0; k<active_size; k++)  //用active_size，还是l?
			{
				iG[t][k] = Q_i[k] * deltaAlpha[i] + Q_j[k] * deltaAlpha[j];
			}
		}
		else{/*这里简化了: 如果只有1路，就不使用中间变量iG[][]*/
			for (int k = 0; k<active_size; k++)
			{
				G[k] += Q_i[k] * deltaAlpha[i] + Q_j[k] * deltaAlpha[j];
			}
		}
		
		if (useShrinkAndCache) {
			int k;
			iG_bar_update_flag[t] = false;
			bool ui = old_is_upper_bound[i];
			bool uj = old_is_upper_bound[j];
			if (ui != is_upper_bound(i))
			{   
				Q_i = this->Q->get_Q(i, l, 1);
				if (ui)
					for (k = 0; k < l; k++) {
						if(nprls > 1)
						  iG_bar[t][k] = -get_C(i) * Q_i[k];
						else
						  G_bar[k] -= get_C(i) * Q_i[k];
					}
					
				else
					for (k = 0; k < l; k++) {
						if(nprls > 1)
							iG_bar[t][k] = get_C(i) * Q_i[k];
						else
							G_bar[k] += get_C(i) * Q_i[k];
					}
				iG_bar_update_flag[t] = true;	
			}

			if (uj != is_upper_bound(j))
			{   
				Q_j = this->Q->get_Q(j, l, 1);
				if(uj) {
					//G_bar[k] -= C_j * Q_j[k];
                    if(iG_bar_update_flag[t])
						for (k = 0; k < l; k++) {
							if(nprls > 1)
								iG_bar[t][k] -= get_C(j) * Q_j[k];
							else
								G_bar[k] -= get_C(j) * Q_j[k];
						}
							
					else
						for (k = 0; k < l; k++) {
							if(nprls > 1)
								iG_bar[t][k] = -get_C(j) * Q_j[k];
							else
								G_bar[k] -= get_C(j) * Q_j[k];
						}
							
				}
				else {//G_bar[k] += C_j * Q_j[k];
					if (iG_bar_update_flag[t])
						for (k = 0; k < l; k++) {
							if(nprls > 1)
							   iG_bar[t][k] += get_C(j) * Q_j[k];
							else
							   G_bar[k] += get_C(j) * Q_j[k];
						}
							
					else
						for (k = 0; k < l; k++) {
							if(nprls > 1)
							   iG_bar[t][k] = get_C(j) * Q_j[k];
							else
							   G_bar[k] += get_C(j) * Q_j[k];
						}		
				}
				iG_bar_update_flag[t] = true;
			}
		 }

		 if (!useShrinkAndCache) {
			delete[] Q_i;
			delete[] Q_j;
		 }
		 QueryPerformanceCounter(&litmp);
		 LONGLONG  updateG_end  = litmp.QuadPart;
		 double temp_updateG_time = (double)(updateG_end - updateG_start) / dfFreq;
		 updateG_time = max(updateG_time, temp_updateG_time);

		 if (printLog) {
			 info("(%d, %d)\n", i, j);
			 fprintf(fout, "(%d, %d)\n", i, j);
		 }
	}

	if (nprls > 1)
	{
		QueryPerformanceCounter(&litmp);
		LONGLONG updateG_start2 = litmp.QuadPart;

		if (useShrinkAndCache) {
			int iG_bar_start_idx = -1;
			for (int t = 0; t < nprls; t++) {
				if (iG_bar_update_flag[t] == true) {
					iG_bar_start_idx = t;
					break;
				}
			}
			if (iG_bar_start_idx != -1) {
				for (int t = iG_bar_start_idx + 1; t < nprls; t++)
					if(iG_bar_update_flag[t] == true)
						for (int k = 0; k < l; k++)
							iG_bar[iG_bar_start_idx][k] += iG_bar[t][k];
				for (int k = 0; k < l; k++)
					G_bar[k] += iG_bar[iG_bar_start_idx][k];
			}

		}
		for (int t = 0; t < nprls; t++)
			for (int z = 0; z < active_size; z++)
				G[z] += iG[t][z];

		QueryPerformanceCounter(&litmp);
		LONGLONG updateG_end2 = litmp.QuadPart;
		double updateG_time2 = (double)(updateG_end2 - updateG_start2) / dfFreq;
		updateG_time += updateG_time2;
	}

	total_updateG_time += updateG_time;
	if (printLog) {

		double v1 = 0;
		for (int z = 0; z<l; z++)
			v1 += alpha[z] * (G[z] + p[z]);
		double obj = v1 / 2;
		info("iter: %d,target_obj: %f\n", iter, obj);
		fprintf(fout_obj, "(%d, %f)\n", iter, obj);
		fprintf(fout_flag, "(%d, %d)\n", iter, nprls);
		fprintf(fout, ", update_G_time=%f, total_updateG_time=%f\n", updateG_time, total_updateG_time);
	}
}
void Solver::update_alpha(int i, int j, double deltaGi, double deltaGj) {

	double C_i = get_C(i);
	double C_j = get_C(j);

	double old_alpha_i = alpha[i];
	double old_alpha_j = alpha[j];

	double Q_ij = (this->Q)->get_Q_ij(i,j);

	if (y[i] != y[j])
	{
		double quad_coef = QD[i] + QD[j] + 2 * Q_ij;
		if (quad_coef <= 0)
			quad_coef = TAU;
		double delta = (-G[i] - deltaGi - G[j] - deltaGj) / quad_coef;
		double diff = alpha[i] - alpha[j];
		alpha[i] += delta;
		alpha[j] += delta;
		if (diff > 0)
		{
			if (alpha[j] < 0)
			{
				alpha[j] = 0;
				alpha[i] = diff;
			}
		}
		else
		{
			if (alpha[i] < 0)
			{
				alpha[i] = 0;
				alpha[j] = -diff;
			}
		}
		if (diff > C_i - C_j)
		{
			if (alpha[i] > C_i)
			{
				alpha[i] = C_i;
				alpha[j] = C_i - diff;
			}
		}
		else
		{
			if (alpha[j] > C_j)
			{
				alpha[j] = C_j;
				alpha[i] = C_j + diff;
			}
		}
	} //(y[i]!=y[j])
	else
	{
		double quad_coef = QD[i] + QD[j] - 2 * Q_ij;
		if (quad_coef <= 0)
			quad_coef = TAU;
		double delta = (G[i]+ deltaGi - G[j]- deltaGj) / quad_coef;
		double sum = alpha[i] + alpha[j];
		alpha[i] -= delta;
		alpha[j] += delta;

		if (sum > C_i)
		{
			if (alpha[i] > C_i)
			{
				alpha[i] = C_i;
				alpha[j] = sum - C_i;
			}
		}
		else
		{
			if (alpha[j] < 0)
			{
				alpha[j] = 0;
				alpha[i] = sum;
			}
		}
		if (sum > C_j)
		{
			if (alpha[j] > C_j)
			{
				alpha[j] = C_j;
				alpha[i] = sum - C_j;
			}
		}
		else
		{
			if (alpha[i] < 0)
			{
				alpha[i] = 0;
				alpha[j] = sum;
			}
		}
	}//(y[i]==y[j])

	deltaAlpha[i] = alpha[i] - old_alpha_i;
	deltaAlpha[j] = alpha[j] - old_alpha_j;

	if (useShrinkAndCache) {
		old_is_upper_bound[i] = is_upper_bound(i);
		old_is_upper_bound[j] = is_upper_bound(j);
	}

	/*移到update_G中执行*/
	//update_alpha_status(i);
	//update_alpha_status(j);

}


void Solver::select_k_ilow_candidates(int parallel_idx, int cached_flag)
{       
	    int i = iup[parallel_idx];

		/*attention*/
		if (cached_flag && !Q->is_cached(i, active_size)) {
			iup[parallel_idx] = -1;
			return;
		}

	    int len = ilowCandidateSize;
		double Gmax = -INF;
		double Gmax2 = -INF;
		int Gmin_idx = -1;

		for (int z = 0; z<len; z++) {
			obj_diff_min[parallel_idx][z] = -INF; 
			deltaG[parallel_idx][z] = 0.0;
			imin[parallel_idx][z]=-1;//由于"grad_diff > eps"等条件，对iup[t]，其候选的Ilow不一定有len个!
		}

		Gmax = (-1)*y[i] * G[i]; //对应i_up是 -y_i*grad(f)_i
		
		const Qfloat *Q_i = NULL;
		if (i != -1) {
			if(!useShrinkAndCache)
			   Q_i = (this->Q)->get_Q(i, active_size); /*non-cache*/
			else
			   Q_i = (this->Q)->get_Q(i, active_size,1); /*cache, 1标识统计命中次数*/
		}
		
		for (int j = 0; j<active_size; j++)
		{
			if (y[j] == +1)
			{
				if (!is_lower_bound(j))//(>0,1):Ilow
				{
					double grad_diff = Gmax + G[j];
					if (G[j] >= Gmax2)
					{
						Gmax2 = G[j];
					}
					if (grad_diff > my_eps)  //注: 这里使用了my_eps
					{   
						double quad_coef = QD[i] + QD[j] - 2.0*y[i] * Q_i[j];
						if (quad_coef<=0) quad_coef = TAU;
						
						double di;
						if (useWSS35) {
							double temp_value = grad_diff / quad_coef;
							di = temp_value * temp_value * grad_diff;
						}
						else
							di = grad_diff * grad_diff / quad_coef;
                        
						//double di = grad_diff;
						if (di >= obj_diff_min[parallel_idx][len - 1])  //用数组obj_diff_min存abs(di)
						{
							int z = len - 1;// 给每个iup留ncpu个候选ilow.
							while (z >= 0 && di >= obj_diff_min[parallel_idx][z]) z--; //把<改成<=是为了更好的和libsvm选集方法靠近
							if (z<len - 1)
							{ //说明z有向前移动: 新值需插入z+1的位置，原来[z+1,ncpu-2]应往后一位

								for (int k = len - 2; k >= z + 1; k--) {//[j+1,ncpu-2]的值往后移一格
									
									imin[parallel_idx][k + 1] = imin[parallel_idx][k];
									obj_diff_min[parallel_idx][k + 1] = obj_diff_min[parallel_idx][k];

								}
								imin[parallel_idx][z + 1] = j;
								obj_diff_min[parallel_idx][z + 1] = di;
							}

						}
						

					}
				}
			}
			else//y[j]==-1
			{
				if (!is_upper_bound(j))//(<C,-1):Ilow
				{
					double grad_diff = Gmax - G[j];
					if (-G[j] >= Gmax2)
					{
						Gmax2 = -G[j];
					}
					if (grad_diff > my_eps)
					{   
						double quad_coef = QD[i] + QD[j] + 2.0*y[i] * Q_i[j];
						if (quad_coef<=0) quad_coef = TAU;
						
						double di;
						if (useWSS35) {
							double temp_value = grad_diff / quad_coef;
							di = temp_value * temp_value * grad_diff;
						}
						else
							di = grad_diff * grad_diff / quad_coef;
                        
						//double di = grad_diff;
						if (di >= obj_diff_min[parallel_idx][len - 1])  //用数组obj_diff_min存abs(di)
						{
							int z = len - 1;// 给每个iup留ncpu个候选ilow.
							while (z >= 0 && di >= obj_diff_min[parallel_idx][z]) z--; //把<改成<=是为了更好的和libsvm选集方法靠近
							if (z<len - 1)
							{ //说明z有向前移动: 新值需插入z+1的位置，原来[z+1,ncpu-2]应往后一位

								for (int k = len - 2; k >= z + 1; k--) {//[j+1,ncpu-2]的值往后移一格
								
									imin[parallel_idx][k + 1] = imin[parallel_idx][k];
									obj_diff_min[parallel_idx][k + 1] = obj_diff_min[parallel_idx][k];
									
								}
								imin[parallel_idx][z + 1] = j;
								obj_diff_min[parallel_idx][z + 1] = di;
							}

						}
					}
					
				}
			}//else
		}//for...
		if (Gmax + Gmax2 < eps)
			iup[parallel_idx] = -1; // notify invalid.

	    if(!useShrinkAndCache) /*non-cache, 手动释放资源*/
          delete[] Q_i; 
}

int Solver::select_kmax_Iup(int parallels) {

	
	//只取Iup的前ncpu个最大的-Fi
	
	for (int t = 0; t<ncpu; t++) {
		nFiup[t] = -INF;
	}

	for (int t = 0; t<active_size; t++)
	{

		if ((y[t] == +1 && !is_upper_bound(t)) || (y[t] == -1 && !is_lower_bound(t)))// Iup	
		{
			int j = ncpu - 1;
			double nf = (-1)*y[t] * G[t];
			while (j >= 0 && nf >= nFiup[j]) j--; //把<改成<=是为了更好的和libsvm选集方法靠近
			if (j<ncpu - 1) { //[j+1,ncpu-2]向后移一位，新的值插入到[j+1]的位置

				for (int i = ncpu - 2; i>j; i--) {//[j+1,ncpu-2]的值往后移一格
					nFiup[i + 1] = nFiup[i];
					iup[i + 1] = iup[i];//索引也要
				}

				nFiup[j + 1] = nf;
				iup[j + 1] = t;
			}
		}
	}
	//对小容量的样本集，有可能找不出ncpu个候选的Iup
	len = ncpu;
	while (nFiup[len - 1] == -INF)
	{
		len--;
	}
	return len;
}

void::Solver::Solve3(int l, const QMatrix& Q, const double *p_, const schar *y_,
	double *alpha_, double Cp, double Cn, double eps,
	SolutionInfo* si) {

	double initial_time = 0.0;
	double free_solve_time = 0.0;
	double total_shrink_time = 0.0;
	double total_regradient_time = 0.0;
	double total_sel_pair_time = 0.0;
	double total_updateG_time = 0.0;

	/*初始化*/
	solve3_init_func(l, Q, p_, y_,
		alpha_, Cp, Cn, eps, initial_time);

	/****************************************************************************************/
	/*******    parallel        ************************************************************/
	/*optimization step*/ 

	unshrink = false; /*标识在采用shrinking且Mup(ak) <= mlow(ak) + 10eps时是否需要重构训练集*/
	int max_iter = max(10000000, l>INT_MAX / 100 ? INT_MAX : 100 * l);

	/*2017-12-22*/
	int counter;
	if(useMyCounter)
		counter = min(l / parallelSize, 1000 / parallelSize) + 1;
	else
        counter = min(l, 1000) + 1;
	
	LARGE_INTEGER litmp;
	QueryPerformanceFrequency(&litmp);
	double dfFreq = (double)litmp.QuadPart;// 获得计数器的时钟频率 

	while (iter < max_iter) 
	{   

		if (useShrinkAndCache) {
			if (--counter == 0)
			{   
				QueryPerformanceCounter(&litmp);
				LONGLONG shrink_start = litmp.QuadPart;
	
				counter = min(l, 1000);
				do_shrinking();
				info(".");

				QueryPerformanceCounter(&litmp);
				LONGLONG shrink_end = litmp.QuadPart;
				double shrink_time = (double)(shrink_end - shrink_start) / dfFreq;
				total_shrink_time += shrink_time;
			}
		}
		int nprls;
		/*选择多路违反对, 内部串行和并行混合进行*/
		nprls = this->select_k_working_set(total_sel_pair_time);
		
		if (nprls == 0)
		{   
			/*若使用了shrinking, 还需要再次验证是否真达到收敛*/
			if (useShrinkAndCache) {

				QueryPerformanceCounter(&litmp);
				LONGLONG regradient_start = litmp.QuadPart;

				// reconstruct the whole gradient
				reconstruct_gradient();
				// reset active set size and check
				active_size = l;
				info("*");

				QueryPerformanceCounter(&litmp);
				LONGLONG regradient_end = litmp.QuadPart;
				double regradient_time = (double)(regradient_end - regradient_start) / dfFreq;
				total_regradient_time += regradient_time;

				nprls = this->select_k_working_set(total_sel_pair_time);
				
				if (nprls == 0)
					break;
				else
					counter = 1;	// do shrinking next iteration
			}
			break; 
		}
		if (nprls != 0) {
			this->iter++;
			if(parallel_algorithm == IIPSMO)
				/*更新多路违反对对全局梯度值的影响*/
				iipsmo_update_G(nprls, total_updateG_time);
			else /*IPSMO/PSMO*/
				ipsmo_update_G(nprls, total_updateG_time);
		}
		  
		
	}//end while(iter < max_iter)
	
	solve3_free_func(max_iter, alpha_, si, free_solve_time);
	total_iter += iter;

	multi_classTime += initial_time + total_sel_pair_time + total_shrink_time + total_regradient_time + total_updateG_time + free_solve_time;

	if (printLog) {
		fprintf(fout, "total_sel_pair_time: %fs\n", total_sel_pair_time);
		fprintf(fout, "total_updateG_time: %fs\n", total_updateG_time);
		fprintf(fout, "initial_time: %fs\n", initial_time);
		fprintf(fout, "free_solve_time: %fs\n", free_solve_time);
		fprintf(fout, "total_shrink_time: %fs\n", total_shrink_time);
		fprintf(fout, "total_regradient_time: %fs\n", total_regradient_time);
		fprintf(fout, "Current Multi class Time : %fs\n", multi_classTime);
	}

	if(useShrinkAndCache){
		//收集cache信息
		long Visits = 0;
		long cacheHits = 0;
		long cacheElements = 0;
		Q.getHitInfo(Visits, cacheHits, cacheElements);
		totalVisits += Visits;
		totalCacheHits += cacheHits;
		totalCacheElements += cacheElements;
	}
	info("\noptimization finished, #iter = %d\n", iter);

}

void Solver::swap_index(int i, int j)
{
	Q->swap_index(i,j);
	swap(y[i],y[j]);
	swap(G[i],G[j]);
	swap(alpha_status[i],alpha_status[j]);
	swap(alpha[i],alpha[j]);
	swap(p[i],p[j]);
	swap(active_set[i],active_set[j]);
	swap(G_bar[i],G_bar[j]);
}

void Solver::reconstruct_gradient()
{
	// reconstruct inactive elements of G from G_bar and free variables

	/*测试用
	int flag;
	int Qi_times = 0;
	int Gi_times = 0;
	*/

	if(active_size == l) return;

	int i,j;
	int nr_free = 0;

	for(j=active_size;j<l;j++)
		G[j] = G_bar[j] + p[j];

	for(j=0;j<active_size;j++)
		if(is_free(j))
			nr_free++;

	if(2*nr_free < active_size)
		info("\nWARNING: using -h 0 may be faster\n");

	if (nr_free*l > 2*active_size*(l-active_size))
	{   
		//flag = 0;
		for(i=active_size;i<l;i++)
		{
			const Qfloat *Q_i = Q->get_Q(i,active_size, 1); /*注: 这里使用cache的*/
			/*测试用*/
			//Qi_times++;
			for(j=0;j<active_size;j++)
				if (is_free(j)) {
					G[i] += alpha[j] * Q_i[j];
				    /*测试用*/
					//Gi_times++;
				}

		}
	}
	else
	{   
		//flag = 1;
		for(i=0;i<active_size;i++)
			if(is_free(i))
			{
				const Qfloat *Q_i = Q->get_Q(i,l, 1);
				/*测试用Qi_times++;*/
				double alpha_i = alpha[i];
				for (j = active_size; j < l; j++) {
					G[j] += alpha_i * Q_i[j];
					/*测试用Gi_times++;*/
				}
			}
	}
	//fprintf(fout, "regident->(%d, %d, %d)", flag, Qi_times, Gi_times);
}
//shrinking+cache
void Solver::Solve(int l, const QMatrix& Q, const double *p_, const schar *y_,
		   double *alpha_, double Cp, double Cn, double eps,
		   SolutionInfo* si, int shrinking)
{   
    /************************************************************************************/
	/**************** initial step **********************/
	
	LARGE_INTEGER litmp; 
    double dfFreq; 
    QueryPerformanceFrequency(&litmp);
    dfFreq = (double)litmp.QuadPart;// 获得计数器的时钟频率 
	QueryPerformanceCounter(&litmp);
	LONGLONG initial_start=litmp.QuadPart;


	//shrinking=0;
    //clock_t initial_start=clock();
	this->old_obj=0; //++

	this->l = l;
	this->Q = &Q;
	QD=Q.get_QD();
	clone(p, p_,l);
	clone(y, y_,l);
	clone(alpha,alpha_,l);
	this->Cp = Cp;
	this->Cn = Cn;
	this->eps = eps;
	unshrink = false;

	// initialize alpha_status
	{
		alpha_status = new char[l];
		for(int i=0;i<l;i++)
			update_alpha_status(i);
	}

	// initialize active set (for shrinking)
	{
		active_set = new int[l];
		for(int i=0;i<l;i++)
			active_set[i] = i;
		active_size = l;
	}

	// initialize gradient
	{
		G = new double[l];
		/*G_bar[i]: 当使用shrinking时，缓存active_size中处于UPPER_BOUND状态的样本点对样本点i的梯度值影响,
		  当需要将i纳入active_size范围并重构其梯度值时可以达到快速计算
		*/
		G_bar = new double[l];
		int i;
		for(i=0;i<l;i++)
		{
			G[i] = p[i];  /*p[i] 一般取 -1， 详见梯度公式*/
			G_bar[i] = 0;
		}
		for(i=0;i<l;i++)
			if(!is_lower_bound(i))
			{
				//const Qfloat *Q_i = Q.get_Q(i,l);
				//0：因这里还没开始正式训练，故不统计cache次数，但对训练时的cache应该是有影响的
				const Qfloat *Q_i = Q.get_Q(i,l,0);
				double alpha_i = alpha[i];
				int j;
				for(j=0;j<l;j++)
					G[j] += alpha_i*Q_i[j];
				
				if(is_upper_bound(i))
					for(j=0;j<l;j++)
						G_bar[j] += get_C(i) * Q_i[j];
                
			}
	}
	  // optimization step
	 int iter = 0;
	 int max_iter = max(10000000, l>INT_MAX/100 ? INT_MAX : 100*l);
	 int counter = min(l,1000)+1;
	
	 QueryPerformanceCounter(&litmp);
	 LONGLONG initial_end=litmp.QuadPart;
	 double initial_time=(double)(initial_end-initial_start)/dfFreq;

     multi_classTime+=initial_time;
	/**************************************************************************************************/
	/************************** iteration phase**********************************/
	
	LONGLONG  sel_set_start;
	LONGLONG  sel_set_end;
	double  sel_set_time;
	double total_sel_time=0;
	double 	total_update_Alpha_G_time=0;

	while(iter < max_iter)
	{
		// show progress and do shrinking
			//int i,j;
		//sel_set_start=clock();
		QueryPerformanceCounter(&litmp);
	    sel_set_start=litmp.QuadPart;
		
		if(--counter == 0)
		{
			counter = min(l,1000);
			if(shrinking) do_shrinking(); 
			info(".");
		}
		int i,j;
		if (useWSS35) {

		    if(wss35_select_working_set(i,j)!=0)
			{   
				/*在使用shrinking的情况下，当找不出违反对时，需要重构active_set集并再次寻找*/
				// reconstruct the whole gradient
				reconstruct_gradient();
				// reset active set size and check
				active_size = l;
				info("*");
				
			    if(wss35_select_working_set(i,j)!=0)  //确定实现收敛
					break;
				else
					counter = 1;	// do shrinking next iteration
			}
		}else{//默认使用LIBSVM
			if (libsvm_select_working_set(i, j) != 0)
			{
				// reconstruct the whole gradient
				reconstruct_gradient();
				// reset active set size and check
				active_size = l;
				info("*");
				if (libsvm_select_working_set(i, j) != 0)
					break;
				else
					counter = 1;	// do shrinking next iteration
			}
		}
		
		
		/*
		int i,j;
		if(select_working_set2(i,j)!=0)
			break;
		*/
		QueryPerformanceCounter(&litmp);
	    sel_set_end=litmp.QuadPart;
		sel_set_time=(double)(sel_set_end-sel_set_start)/dfFreq;
		//sel_set_end=clock();
		//sel_set_time=(double)(sel_set_end-sel_set_start)/CLOCKS_PER_SEC;

		++iter;
		// update alpha[i] and alpha[j], handle bounds carefully

		//clock_t iter_start=clock();
		QueryPerformanceCounter(&litmp);
	    LONGLONG iter_start=litmp.QuadPart;



		//const Qfloat *Q_i = Q.get_Q(i,active_size);
		//const Qfloat *Q_j = Q.get_Q(j,active_size);

		const Qfloat *Q_i = Q.get_Q(i,active_size,0); //此处对i不统计cache，因为前面已统计过
		const Qfloat *Q_j = Q.get_Q(j,active_size,1);

		double C_i = get_C(i);
		double C_j = get_C(j);

		double old_alpha_i = alpha[i];
		double old_alpha_j = alpha[j];
		double delta;
		if(y[i]!=y[j])
		{
			double quad_coef = QD[i]+QD[j]+2*Q_i[j];
			if (quad_coef <= 0)
				quad_coef = TAU;
		    delta = (-G[i]-G[j])/quad_coef;
			double diff = alpha[i] - alpha[j];
			alpha[i] += delta;
			alpha[j] += delta;
			
			if(diff > 0)
			{
				if(alpha[j] < 0)
				{
					alpha[j] = 0;
					alpha[i] = diff;
				}
			}
			else
			{
				if(alpha[i] < 0)
				{
					alpha[i] = 0;
					alpha[j] = -diff;
				}
			}
			if(diff > C_i - C_j)
			{
				if(alpha[i] > C_i)
				{
					alpha[i] = C_i;
					alpha[j] = C_i - diff;
				}
			}
			else
			{
				if(alpha[j] > C_j)
				{
					alpha[j] = C_j;
					alpha[i] = C_j + diff;
				}
			}
		}
		else
		{
			double quad_coef = QD[i]+QD[j]-2*Q_i[j];
			if (quad_coef <= 0)
				quad_coef = TAU;
		    delta = (G[i]-G[j])/quad_coef;
			double sum = alpha[i] + alpha[j];
			alpha[i] -= delta;
			alpha[j] += delta;

			if(sum > C_i)
			{
				if(alpha[i] > C_i)
				{
					alpha[i] = C_i;
					alpha[j] = sum - C_i;
				}
			}
			else
			{
				if(alpha[j] < 0)
				{
					alpha[j] = 0;
					alpha[i] = sum;
				}
			}
			if(sum > C_j)
			{
				if(alpha[j] > C_j)
				{
					alpha[j] = C_j;
					alpha[i] = sum - C_j;
				}
			}
			else
			{
				if(alpha[i] < 0)
				{
					alpha[i] = 0;
					alpha[j] = sum;
				}
			}

		}
		//fprintf(fout,"  iter:%d'th G[%d]=%f,G[%d]=%f,delta=%f\n",iter,i,G[i],j,G[j],delta);
	
		// update G

		double delta_alpha_i = alpha[i] - old_alpha_i;
		double delta_alpha_j = alpha[j] - old_alpha_j;
		
		for(int k=0;k<active_size;k++)
		{
			G[k] += Q_i[k]*delta_alpha_i + Q_j[k]*delta_alpha_j;
		}

		//update_alpha_status(i);
		//update_alpha_status(j);
		// update alpha_status and G_bar
	   {
			bool ui = is_upper_bound(i);
			bool uj = is_upper_bound(j);
			update_alpha_status(i);
			update_alpha_status(j);
			int k;

			/*当有因子现在或上一次处于UPPER_BOUND时，更新其对G_bar数组的影响*/
			if(ui != is_upper_bound(i))
			{
				//Q_i = Q.get_Q(i,l);
				Q_i = Q.get_Q(i,l,0);
				if(ui)
					for(k=0;k<l;k++)
						G_bar[k] -= C_i * Q_i[k];
				else
					for(k=0;k<l;k++)
						G_bar[k] += C_i * Q_i[k];
			}

			if(uj != is_upper_bound(j))
			{
				//Q_j = Q.get_Q(j,l);
				Q_j = Q.get_Q(j,l,0);
				if(uj)
					for(k=0;k<l;k++)
						G_bar[k] -= C_j * Q_j[k];
				else
					for(k=0;k<l;k++)
						G_bar[k] += C_j * Q_j[k];
			}
		}
	
		QueryPerformanceCounter(&litmp);
	    LONGLONG iter_end=litmp.QuadPart;
		double iter_time=(double)(iter_end-iter_start)/dfFreq;

		multi_classTime+=iter_time+sel_set_time;
		total_sel_time+=sel_set_time;
		total_update_Alpha_G_time +=iter_time;
		if(printLog)
		{   
			double v2=0;
			for(int z=0;z<l;z++)       
			   v2 += (alpha[z]) * (G[z]+ p[z]); 
			double obj=v2/2;
			int flag= 1;//并行数
			fprintf(fout_obj,"%f\n",obj);
			fprintf(fout_flag,"%d\n",flag);
			fprintf(fout,"iter:%d i=%d j=%d obj=%.3f sel_set_time=%f; iter_time=%f\n",iter,i,j,obj,sel_set_time,iter_time);	
		}
	    

	}
	
	//收集cache信息
	long Visits=0;
	long cacheHits=0;
	long cacheElements=0;
	Q.getHitInfo(Visits,cacheHits,cacheElements);
    totalVisits+=Visits;
	totalCacheHits+=cacheHits;
	totalCacheElements+=cacheElements;

	//clock_t exit_iter_start=clock();

	QueryPerformanceCounter(&litmp);
	LONGLONG exit_iter_start=litmp.QuadPart;

	if(iter >= max_iter)
	{
		if(active_size < l)
		{
			// reconstruct the whole gradient to calculate objective value
			reconstruct_gradient();
			active_size = l;
			info("*");
		}
		fprintf(stderr,"\nWARNING: reaching max number of iterations\n");
	}

	// calculate rho

	si->rho = calculate_rho();

	// calculate objective value
	{
		double v = 0;
		int i;
		for(i=0;i<l;i++)
			v += alpha[i] * (G[i] + p[i]);

		si->obj = v/2;
	}

	// put back the solution
	{
		for(int i=0;i<l;i++)
			alpha_[active_set[i]] = alpha[i];
	}

	// juggle everything back
	/*{
		for(int i=0;i<l;i++)
			while(active_set[i] != i)
				swap_index(i,active_set[i]);
				// or Q.swap_index(i,active_set[i]);
	}*/

	si->upper_bound_p = Cp;
	si->upper_bound_n = Cn;

	delete[] p;
	delete[] y;
	delete[] alpha;
	delete[] alpha_status;
	delete[] active_set;
	delete[] G;
	delete[] G_bar;

	

	QueryPerformanceCounter(&litmp);
	LONGLONG exit_iter_end=litmp.QuadPart;
	double  exit_iter_time=(double)(exit_iter_end-exit_iter_start)/dfFreq;

	multi_classTime+=exit_iter_time;
	total_iter+=iter; //important.

	fprintf(fout, "total_sel_time: %fs\n", total_sel_time);
	fprintf(fout, "total_update_Alpha_G_time: %fs\n", total_update_Alpha_G_time);
	fprintf(fout, "Current Multi class Time : %fs\n", multi_classTime);

	printf("\ntotal_sel_time: %fs\n", total_sel_time);
	printf("total_update_Alpha_G_time: %fs\n", total_update_Alpha_G_time);
	printf("Current Multi class Time : %fs\n", multi_classTime);

	info("\noptimization finished, #iter = %d\n",iter);
	fprintf(fout,"#iter = %d\n",iter);
}
//non-shrinking+non-cache
void Solver::Solve2(int l, const QMatrix& Q, const double *p_, const schar *y_,
		   double *alpha_, double Cp, double Cn, double eps,
		   SolutionInfo* si)
{   
    /************************************************************************************/
	/**************** initial step **********************/
	//Non-shrinking+ Non-Cache version

	LARGE_INTEGER litmp; 
    double dfFreq; 
    QueryPerformanceFrequency(&litmp);
    dfFreq = (double)litmp.QuadPart;// 获得计数器的时钟频率 
	QueryPerformanceCounter(&litmp);
	LONGLONG initial_start=litmp.QuadPart;


    //clock_t initial_start=clock();
	this->old_obj=0; //++

	this->l = l;
	this->Q = &Q;
	QD=Q.get_QD();
	clone(p, p_,l);
	clone(y, y_,l);
	clone(alpha,alpha_,l);
	this->Cp = Cp;
	this->Cn = Cn;
	this->eps = eps;
	//unshrking=false;

	// initialize alpha_status
	{
		alpha_status = new char[l];
		for(int i=0;i<l;i++)
			update_alpha_status(i);
	}

	// initialize active set (for shrinking)
	{
		active_set = new int[l];
		for(int i=0;i<l;i++)
			active_set[i] = i;
		active_size = l;
	}

	// initialize gradient
	{
		G = new double[l];
		//G_bar = new double[l];
		int i;
		for(i=0;i<l;i++)
		{
			G[i] = p[i];
			//G_bar[i] = 0;
		}
		for(i=0;i<l;i++)
			if(!is_lower_bound(i))
			{
				const Qfloat *Q_i = Q.get_Q(i,l); //由于这里是non-cache版本，可考虑把get_Q改成non-cache版本
				double alpha_i = alpha[i];
				int j;
				for(j=0;j<l;j++)
					G[j] += alpha_i*Q_i[j];
				/*
				if(is_upper_bound(i))
					for(j=0;j<l;j++)
						G_bar[j] += get_C(i) * Q_i[j];
                */
				delete [] Q_i;  //++
			}
	}
	  // optimization step
	 int iter = 0;
	 int max_iter = max(10000000, l>INT_MAX/100 ? INT_MAX : 100*l);
	 int counter = min(l,1000)+1;
	
     //clock_t initial_end=clock();	
	 QueryPerformanceCounter(&litmp);
	 LONGLONG initial_end=litmp.QuadPart;
     //double initial_time=(double)(initial_end-initial_start)/CLOCKS_PER_SEC;  //求工作集前的准备工作
	 double initial_time=(double)(initial_end-initial_start)/dfFreq;

     multi_classTime+=initial_time;
	/**************************************************************************************************/
	/************************** iteration phase**********************************/
	/*
	clock_t  sel_set_start;
	clock_t  sel_set_end;
	double  sel_set_time;
	double total_sel_time=0;
	double 	serialTime=0;
	*/
	LONGLONG  sel_set_start;
	LONGLONG  sel_set_end;
	double  sel_set_time;
	double total_sel_time=0;
	double 	total_update_Alpha_G_time =0;

	while(iter < max_iter)
	{
		// show progress and do shrinking

		QueryPerformanceCounter(&litmp);
	    sel_set_start=litmp.QuadPart;
		
		/*
		if(--counter == 0)
		{
			counter = min(l,1000);
			if(shrinking) do_shrinking();
			info(".");
		}
		int i,j;
		//if(select_working_set(i,j)!=0)
		if(select_working_set6(i,j)!=0)
		{
			// reconstruct the whole gradient
			reconstruct_gradient();
			// reset active set size and check
			active_size = l;
			info("*");
			//if(select_working_set(i,j)!=0)
			if(select_working_set6(i,j)!=0)
				break;
			else
				counter = 1;	// do shrinking next iteration
		}
		*/
		//non-shrinking + non-cache
		int i,j;

		if (useWSS35) {
			if(wss35_select_working_set(i,j)!=0) //new working set selection
			   break;
		}
		else {//默认使用LIBSVM
			if (libsvm_select_working_set(i, j) != 0)    //original libsvm wss.
				break;
		}
		QueryPerformanceCounter(&litmp);
	    sel_set_end=litmp.QuadPart;
		sel_set_time=(double)(sel_set_end-sel_set_start)/dfFreq;

		++iter;
		// update alpha[i] and alpha[j], handle bounds carefully

		
		QueryPerformanceCounter(&litmp);
	    LONGLONG iter_start=litmp.QuadPart;



		const Qfloat *Q_i = Q.get_Q(i,active_size);
		const Qfloat *Q_j = Q.get_Q(j,active_size);

		double C_i = get_C(i);
		double C_j = get_C(j);

		double old_alpha_i = alpha[i];
		double old_alpha_j = alpha[j];
		double delta;
		if(y[i]!=y[j])
		{
			double quad_coef = QD[i]+QD[j]+2*Q_i[j];
			if (quad_coef <= 0)
				quad_coef = TAU;
		    delta = (-G[i]-G[j])/quad_coef;
			double diff = alpha[i] - alpha[j];
			alpha[i] += delta;
			alpha[j] += delta;
			
			if(diff > 0)
			{
				if(alpha[j] < 0)
				{
					alpha[j] = 0;
					alpha[i] = diff;
				}
			}
			else
			{
				if(alpha[i] < 0)
				{
					alpha[i] = 0;
					alpha[j] = -diff;
				}
			}
			if(diff > C_i - C_j)
			{
				if(alpha[i] > C_i)
				{
					alpha[i] = C_i;
					alpha[j] = C_i - diff;
				}
			}
			else
			{
				if(alpha[j] > C_j)
				{
					alpha[j] = C_j;
					alpha[i] = C_j + diff;
				}
			}
		}
		else
		{
			double quad_coef = QD[i]+QD[j]-2*Q_i[j];
			if (quad_coef <= 0)
				quad_coef = TAU;
		    delta = (G[i]-G[j])/quad_coef;
			double sum = alpha[i] + alpha[j];
			alpha[i] -= delta;
			alpha[j] += delta;

			if(sum > C_i)
			{
				if(alpha[i] > C_i)
				{
					alpha[i] = C_i;
					alpha[j] = sum - C_i;
				}
			}
			else
			{
				if(alpha[j] < 0)
				{
					alpha[j] = 0;
					alpha[i] = sum;
				}
			}
			if(sum > C_j)
			{
				if(alpha[j] > C_j)
				{
					alpha[j] = C_j;
					alpha[i] = sum - C_j;
				}
			}
			else
			{
				if(alpha[i] < 0)
				{
					alpha[i] = 0;
					alpha[j] = sum;
				}
			}

		}
		//fprintf(fout,"  iter:%d'th G[%d]=%f,G[%d]=%f,delta=%f\n",iter,i,G[i],j,G[j],delta);
	
		// update G

		double delta_alpha_i = alpha[i] - old_alpha_i;
		double delta_alpha_j = alpha[j] - old_alpha_j;
		
		for(int k=0;k<active_size;k++)
		{
			G[k] += Q_i[k]*delta_alpha_i + Q_j[k]*delta_alpha_j;
		}

		update_alpha_status(i);
		update_alpha_status(j);
		/*
		// update alpha_status and G_bar
	   {
			bool ui = is_upper_bound(i);
			bool uj = is_upper_bound(j);
			update_alpha_status(i);
			update_alpha_status(j);
			int k;
			if(ui != is_upper_bound(i))
			{
				Q_i = Q.get_Q(i,l);
				if(ui)
					for(k=0;k<l;k++)
						G_bar[k] -= C_i * Q_i[k];
				else
					for(k=0;k<l;k++)
						G_bar[k] += C_i * Q_i[k];
			}

			if(uj != is_upper_bound(j))
			{
				Q_j = Q.get_Q(j,l);
				if(uj)
					for(k=0;k<l;k++)
						G_bar[k] -= C_j * Q_j[k];
				else
					for(k=0;k<l;k++)
						G_bar[k] += C_j * Q_j[k];
			}
		}
		*/
		delete [] Q_i;
		delete [] Q_j;

		QueryPerformanceCounter(&litmp);
	    LONGLONG iter_end=litmp.QuadPart;
		double iter_time=(double)(iter_end-iter_start)/dfFreq;

		multi_classTime+=iter_time+sel_set_time;
		total_sel_time+=sel_set_time;
		total_update_Alpha_G_time +=iter_time;
		if(printLog)
		{   
			double v2=0;
			for(int z=0;z<l;z++)       
			   v2 += (alpha[z]) * (G[z]+ p[z]); 
			double obj=v2/2;
			int flag=0;//1表示parallel，0表示single
			fprintf(fout_obj,"%f\n",obj);
			fprintf(fout_flag,"%d\n",flag);
			fprintf(fout,"iter:%d i=%d j=%d obj=%.3f sel_set_time=%f; iter_time=%f\n",iter,i,j,obj,sel_set_time,iter_time);	
		}
	       

	}
	
	QueryPerformanceCounter(&litmp);
	LONGLONG exit_iter_start=litmp.QuadPart;

	if(iter >= max_iter)
	{   /*
		if(active_size < l)
		{
			// reconstruct the whole gradient to calculate objective value
			reconstruct_gradient();
			active_size = l;
			info("*");
		}
		*/
		fprintf(stderr,"\nWARNING: reaching max number of iterations\n");
	}

	// calculate rho

	si->rho = calculate_rho();

	// calculate objective value
	{
		double v = 0;
		int i;
		for(i=0;i<l;i++)
			v += alpha[i] * (G[i] + p[i]);

		si->obj = v/2;
	}

	// put back the solution
	{
		for(int i=0;i<l;i++)
			alpha_[active_set[i]] = alpha[i];
	}

	// juggle everything back
	/*{
		for(int i=0;i<l;i++)
			while(active_set[i] != i)
				swap_index(i,active_set[i]);
				// or Q.swap_index(i,active_set[i]);
	}*/

	si->upper_bound_p = Cp;
	si->upper_bound_n = Cn;

	delete[] p;
	delete[] y;
	delete[] alpha;
	delete[] alpha_status;
	delete[] active_set;
	delete[] G;
	//delete[] G_bar;

	QueryPerformanceCounter(&litmp);
	LONGLONG exit_iter_end=litmp.QuadPart;
	double  exit_iter_time=(double)(exit_iter_end-exit_iter_start)/dfFreq;

	multi_classTime+=exit_iter_time;
	total_iter+=iter; //important.

	info("\noptimization finished, #iter = %d\n",iter);
	fprintf(fout,"#iter = %d\n",iter);
	fprintf(fout, "total_sel_time: %fs\n", total_sel_time);
	fprintf(fout, "total_update_Alpha_G_time: %fs\n", total_update_Alpha_G_time);
	fprintf(fout, "Current Multi class Time : %fs\n", multi_classTime);

}
// return 1 if already optimal, return 0 otherwise
int Solver::libsvm_select_working_set(int &out_i, int &out_j)
{
	// return i,j such that
	// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
	// j: minimizes the decrease of obj value
	//    (if quadratic coefficeint <= 0, replace it with tau)
	//    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)
	double Gmax = -INF;
	double Gmax2 = -INF;
	int Gmax_idx = -1;
	int Gmin_idx = -1;
	double obj_diff_min = INF;

	for(int t=0;t<active_size;t++)
		if(y[t]==+1)	
		{
			if(!is_upper_bound(t))
				if(-G[t] >= Gmax)
				{
					Gmax = -G[t];
					Gmax_idx = t;
				}
		}
		else
		{
			if(!is_lower_bound(t))
				if(G[t] >= Gmax)
				{
					Gmax = G[t];
					Gmax_idx = t;
				}
		}

	int i = Gmax_idx;
	const Qfloat *Q_i = NULL;
	if (i != -1) { // NULL Q_i not accessed: Gmax=-INF if i=-1
		if (useShrinkAndCache) {
			Q_i = Q->get_Q(i, active_size, 1);//cache版函数，且统计cache命中情况
		}
		else {
			Q_i = Q->get_Q(i,active_size);//non-cache版函数，且不统计cache命中情况
		}
	}
	for(int j=0;j<active_size;j++)
	{
		if(y[j]==+1)
		{
			if (!is_lower_bound(j))
			{
				double grad_diff=Gmax+G[j];
				if (G[j] >= Gmax2)
					Gmax2 = G[j];
				if (grad_diff > 0)
				{
					double obj_diff;
					double quad_coef = QD[i]+QD[j]-2.0*y[i]*Q_i[j];
					
					if (quad_coef > 0)
						obj_diff = -(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff = -(grad_diff*grad_diff)/TAU;

					if (obj_diff <= obj_diff_min)
					{
						Gmin_idx=j;
						obj_diff_min = obj_diff;
					}
					
				}
			}
		}
		else
		{
			if (!is_upper_bound(j))
			{
				double grad_diff= Gmax-G[j];
				if (-G[j] >= Gmax2)
					Gmax2 = -G[j];
				if (grad_diff > 0)
				{
					double obj_diff;
					double quad_coef = QD[i]+QD[j]+2.0*y[i]*Q_i[j];
					if (quad_coef > 0)
						obj_diff = -(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff = -(grad_diff*grad_diff)/TAU;
					if (obj_diff <= obj_diff_min)
					{
						Gmin_idx=j;
						obj_diff_min = obj_diff;
					}
					
				}
			}
		}
	}

	if(Gmax+Gmax2 < eps)
		return 1;

	out_i = Gmax_idx;
	out_j = Gmin_idx;
	if(!useShrinkAndCache)
	    delete [] Q_i; //only for non-cache
	return 0;
}
int Solver::wss35_select_working_set(int &out_i, int &out_j)   //new working set selection
{
	// return i,j such that
	// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
	// j: minimizes the decrease of obj value
	//    (if quadratic coefficeint <= 0, replace it with tau)
	//    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)
	double Gmax = -INF;
	double Gmax2 =-INF;  //change
	int Gmax_idx = -1;
	int Gmin_idx = -1;
	double obj_diff_min = INF;
	double di;//++
	double gMaxdi=-INF;//++
	
	for(int t=0;t<active_size;t++)
		if(y[t]==+1)	
		{
			if(!is_upper_bound(t))
				if(-G[t] >= Gmax)
				{
					Gmax = -G[t];
					Gmax_idx = t;
				}
		}
		else
		{
			if(!is_lower_bound(t))
				if(G[t] >= Gmax)
				{
					Gmax = G[t];
					Gmax_idx = t;
				}
		}

	int i = Gmax_idx;
	const Qfloat *Q_i = NULL;
	if (i != -1) { // NULL Q_i not accessed: Gmax=-INF if i=-1
		if (useShrinkAndCache) {
			Q_i = Q->get_Q(i, active_size, 1);//cache版函数，且统计cache命中情况
		}
		else {
			Q_i = Q->get_Q(i, active_size);//non-cache版函数，且不统计cache命中情况
		}
	}
	for(int j=0;j<active_size;j++)
	{   
			if(y[j]==+1)
			{
				if (!is_lower_bound(j))
				{
					double grad_diff=Gmax+G[j];
					if (G[j] >= Gmax2)  //沿用libsvm的工作集选择
					    Gmax2 = G[j];
					if (grad_diff > 0)
					{
						double obj_diff;
						double quad_coef = QD[i]+QD[j]-2.0*y[i]*Q_i[j];
						if(quad_coef<0) quad_coef=TAU;
						//di=(grad_diff*grad_diff*grad_diff)/(quad_coef*quad_coef);
						double temp_value = grad_diff / quad_coef;
						di = temp_value * temp_value * grad_diff;
						//di=(grad_diff*grad_diff*grad_diff)/quad_coef;
						//obj_diff = -(grad_diff*grad_diff)/quad_coef;
						//关于di的
						//double di=y[i]*grad_diff/quad_coef; //di=yi*bij/aij ;
						/*
						double Cj = Cp;  //y[j]=1
						if(y[i]==y[j])	
						{
							if(di>0)
								di=((di)<(Ci-alpha[i])?(di<alpha[j]?di:alpha[j]):(Ci-alpha[i]<alpha[j]?Ci-alpha[i]:alpha[j])); 
							else
								di=((di)>(-alpha[i])?(di>alpha[j]-Cj?di:(alpha[j]-Cj)):(-alpha[i]>(alpha[j]-Cj)?-alpha[i]:(alpha[j]-Cj)));
						}
						else
						{
							if(di>0)
								di=((di)<(Ci-alpha[i])?(di<Cj-alpha[j]?di:Cj-alpha[j]):(Ci-alpha[i]<Cj-alpha[j]?Ci-alpha[i]:Cj-alpha[j])); 
							else
								di=((di)>(-alpha[i])?((di)>(-alpha[j])?(di):(-alpha[j])):((-alpha[i])>(-alpha[j])?(-alpha[i]):(-alpha[j]))); 
						}
						*/
						//di=di*obj_diff;
						//if(di<0) di=-di;
						if(di>=gMaxdi)
						{
							gMaxdi=di;
							Gmin_idx=j;
						}
				    
					}
				}
			}
			else 
			{
				if (!is_upper_bound(j))
				{
					double grad_diff= Gmax-G[j];
					if (-G[j] >= Gmax2)
					    Gmax2 = -G[j];
					if (grad_diff > 0)
					{
						double obj_diff;
						double quad_coef = QD[i]+QD[j]+2.0*y[i]*Q_i[j];
						if(quad_coef<0) quad_coef=TAU;
						//di=(grad_diff*grad_diff*grad_diff)/(quad_coef*quad_coef);
						/*
						  libsvm: grad_diff*grad_diff / quad_coef
						  wss35: grad_diff*grad_diff*grad_diff /(quad_coef*quad_coef)
						*/
						double temp_value = grad_diff / quad_coef;
						di = temp_value * temp_value * grad_diff;
						//di=(grad_diff*grad_diff*grad_diff)/quad_coef;
						//obj_diff = -(grad_diff*grad_diff)/quad_coef;
						//double di=y[i]*grad_diff/quad_coef; //di=yi*bij/aij ;
						/*
						obj_diff = -(grad_diff*grad_diff)/quad_coef;
						di= y[i]*grad_diff/quad_coef; //di=yi*bij/aij ;
						/*
						double Cj = Cn;  //y[j]=-1
						if(y[i]==y[j])	
						{
							if(di>0)
								di=((di)<(Ci-alpha[i])?(di<alpha[j]?di:alpha[j]):(Ci-alpha[i]<alpha[j]?Ci-alpha[i]:alpha[j])); 
							else
								di=((di)>(-alpha[i])?(di>alpha[j]-Cj?di:(alpha[j]-Cj)):(-alpha[i]>(alpha[j]-Cj)?-alpha[i]:(alpha[j]-Cj)));
						}
						else
						{
							if(di>0)
								di=((di)<(Ci-alpha[i])?(di<Cj-alpha[j]?di:Cj-alpha[j]):(Ci-alpha[i]<Cj-alpha[j]?Ci-alpha[i]:Cj-alpha[j])); 
							else
								di=((di)>(-alpha[i])?((di)>(-alpha[j])?(di):(-alpha[j])):((-alpha[i])>(-alpha[j])?(-alpha[i]):(-alpha[j]))); 
						}
						*/	
						//di=di*obj_diff;//联合
						//if(di<0) di=-di;
						if(di>=gMaxdi)
						{
							gMaxdi=di;
							Gmin_idx=j;
						}

					}
				}
			}
		}

	if(Gmax+Gmax2 < eps)
		return 1;
	out_i = Gmax_idx;
	out_j = Gmin_idx;
	if(!useShrinkAndCache)
	   delete [] Q_i;  //for non-shrinking non-cache
	return 0;
}
int Solver::select_working_set3(int &out_i, int &out_j, double& total_sel_pair_time)   //new working set selection
{   

	LARGE_INTEGER litmp;
	QueryPerformanceFrequency(&litmp);
	double dfFreq = (double)litmp.QuadPart;// 获得计数器的时钟频率 
	QueryPerformanceCounter(&litmp);
	LONGLONG sel_pair_start = litmp.QuadPart;

	double Gmax = -INF;
	double Gmax2 = -INF;  //change
	int Gmax_idx = -1;
	int Gmin_idx = -1;
	double obj_diff_min = INF;
	double gMaxdi = -INF;//++

	for (int t = 0; t<active_size; t++)
		if (y[t] == +1)
		{
			if (!is_upper_bound(t))
				if (-G[t] >= Gmax)
				{
					Gmax = -G[t];
					Gmax_idx = t;
				}
		}
		else
		{
			if (!is_lower_bound(t))
				if (G[t] >= Gmax)
				{
					Gmax = G[t];
					Gmax_idx = t;
				}
		}

	int i = Gmax_idx;
	const Qfloat *Q_i = NULL;
	if (i != -1) { // NULL Q_i not accessed: Gmax=-INF if i=-1
		if (useShrinkAndCache) {
			Q_i = Q->get_Q(i, active_size, 1);//cache版函数，且统计cache命中情况
		}
		else {
			Q_i = Q->get_Q(i, active_size);//non-cache版函数，且不统计cache命中情况
		}
	}
	for (int j = 0; j<active_size; j++)
	{
		if (y[j] == +1)
		{
			if (!is_lower_bound(j))
			{
				double grad_diff = Gmax + G[j];
				if (G[j] >= Gmax2)  //沿用libsvm的工作集选择
					Gmax2 = G[j];
				if (grad_diff > 0)
				{
					double obj_diff;
					double quad_coef = QD[i] + QD[j] - 2.0*y[i] * Q_i[j];
					if (quad_coef<0) quad_coef = TAU;

					double di;
					if (useWSS35) {
						double temp_value = grad_diff / quad_coef;
						di = temp_value * temp_value * grad_diff;
					}
					else
					    di = grad_diff * grad_diff / quad_coef;

					if (di >= gMaxdi)
					{
						gMaxdi = di;
						Gmin_idx = j;
					}

				}
			}
		}
		else //y[j]==-1
		{
			if (!is_upper_bound(j))
			{
				double grad_diff = Gmax - G[j];
				if (-G[j] >= Gmax2)
					Gmax2 = -G[j];
				if (grad_diff > 0)
				{
					double obj_diff;
					double quad_coef = QD[i] + QD[j] + 2.0*y[i] * Q_i[j];
					if (quad_coef<0) quad_coef = TAU;

					double di;
					if (useWSS35) {
						double temp_value = grad_diff / quad_coef;
						di = temp_value * temp_value * grad_diff;
					}else
						di = grad_diff * grad_diff / quad_coef;

					if (di >= gMaxdi)
					{
						gMaxdi = di;
						Gmin_idx = j;
					}

				}
			}
		}
	}

	if (Gmax + Gmax2 < eps)
		return 0;
	out_i = Gmax_idx;
	out_j = Gmin_idx;
	this->update_alpha(out_i, out_j, 0.0, 0.0); /*important*/

	if (!useShrinkAndCache)
		delete[] Q_i;  //for non-shrinking non-cache

	QueryPerformanceCounter(&litmp);
	LONGLONG sel_pair_end = litmp.QuadPart;
	double sel_pair_time = (double)(sel_pair_end - sel_pair_start) / dfFreq;
	total_sel_pair_time += sel_pair_time;
	if (printLog) {
		fprintf(fout, "iter=%d, nprls = %d, sel_pair_time=%f, total_sel_pair_time=%f", iter, 1, sel_pair_time, total_sel_pair_time);
	}
	return 1;
}


bool Solver::be_shrunk(int i, double Gmax1, double Gmax2)
{   
	/*shrunk条件详见本论文P46*/
	if(is_upper_bound(i))
	{
		if(y[i]==+1)  //(C,+1) and (0,-1)属于Ilow,与Gmax1 比.(0,+1) and (C,-1)属于Iup，与Gmax2比.
			return(-G[i] > Gmax1);
		else
			return(-G[i] > Gmax2);
	}
	else if(is_lower_bound(i))
	{
		if(y[i]==+1)
			return(G[i] > Gmax2);
		else	
			return(G[i] > Gmax1);
	}
	else
		return(false);
}

void Solver::do_shrinking()
{   
	int i;

	/*测试用
	shrink_times++;
	int old_active_size = active_size;
	double regradient_time = 0.0;
	int swap_counter = 0;
	*/

	/* 这里的Gmax1、Gmax2 分别对应本论文P45中的
	   Mup(ak)和 mlow(ak)
	*/
	double Gmax1 = -INF;		// max { -y_i * grad(f)_i | i in I_up(\alpha) }
	double Gmax2 = -INF;		// max { y_i * grad(f)_i | i in I_low(\alpha) }

	// find maximal violating pair first
	for(i=0;i<active_size;i++)
	{
		if(y[i]==+1)	
		{
			if(!is_upper_bound(i))	
			{
				if(-G[i] >= Gmax1)
					Gmax1 = -G[i];
			}
			if(!is_lower_bound(i))	
			{
				if(G[i] >= Gmax2)
					Gmax2 = G[i];
			}
		}
		else	
		{
			if(!is_upper_bound(i))	
			{
				if(-G[i] >= Gmax2)
					Gmax2 = -G[i];
			}
			if(!is_lower_bound(i))	
			{
				if(G[i] >= Gmax1)
					Gmax1 = G[i];
			}
		}
	}
	
	if(unshrink == false && Gmax1 + Gmax2 <= eps*10) 
	{   
		/*
		LARGE_INTEGER litmp;
		QueryPerformanceFrequency(&litmp);
		double dfFreq = (double)litmp.QuadPart;// 获得计数器的时钟频率 
		QueryPerformanceCounter(&litmp);
		LONGLONG regradient_start = litmp.QuadPart;
		*/
		unshrink = true;
		reconstruct_gradient();
		active_size = l;
		info("*");
		/*
		QueryPerformanceCounter(&litmp);
		LONGLONG regradient_end = litmp.QuadPart;
	    regradient_time = (double)(regradient_end - regradient_start) / dfFreq;
		*/
	}
	
	for(i=0;i<active_size;i++)
		if (be_shrunk(i, Gmax1, Gmax2))
		{
			active_size--; //因为active_size个样本，最高索引是active_size-1.
			while (active_size > i)//当active_size从右移到==i时，整个for循环结束此时包含在active_size的是索引[0,1,...i-1]
			{
				if (!be_shrunk(active_size, Gmax1, Gmax2))
				{
					swap_index(i,active_size);
					//swap_counter++;
					break;
				}
				active_size--;
			}
		}

	//printf("%d: %d\n",count,active_size);
	//fprintf(fout, "shrink: (%d, %d, %d, %d, %f)", shrink_times, old_active_size, active_size, swap_counter, regradient_time);
}

double Solver::calculate_rho()
{
	double r;
	int nr_free = 0;
	double ub = INF, lb = -INF, sum_free = 0;
	for(int i=0;i<active_size;i++)
	{
		double yG = y[i]*G[i];

		if(is_upper_bound(i))
		{
			if(y[i]==-1)
				ub = min(ub,yG);
			else
				lb = max(lb,yG);
		}
		else if(is_lower_bound(i))
		{
			if(y[i]==+1)
				ub = min(ub,yG);
			else
				lb = max(lb,yG);
		}
		else
		{
			++nr_free;
			sum_free += yG;
		}
	}

	if(nr_free>0)
		r = sum_free/nr_free;
	else
		r = (ub+lb)/2;

	return r;
}

//
// Solver for nu-svm classification and regression
//
// additional constraint: e^T \alpha = constant
//
class Solver_NU: public Solver
{
public:
	Solver_NU() {}
	void Solve(int l, const QMatrix& Q, const double *p, const schar *y,
		   double *alpha, double Cp, double Cn, double eps,
		   SolutionInfo* si, int shrinking)
	{
		this->si = si;
		Solver::Solve(l,Q,p,y,alpha,Cp,Cn,eps,si,shrinking);
	}
private:
	SolutionInfo *si;
	int select_working_set(int &i, int &j);
	double calculate_rho();
	bool be_shrunk(int i, double Gmax1, double Gmax2, double Gmax3, double Gmax4);
	void do_shrinking();
};

// return 1 if already optimal, return 0 otherwise
int Solver_NU::select_working_set(int &out_i, int &out_j)
{
	// return i,j such that y_i = y_j and
	// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
	// j: minimizes the decrease of obj value
	//    (if quadratic coefficeint <= 0, replace it with tau)
	//    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

	double Gmaxp = -INF;
	double Gmaxp2 = -INF;
	int Gmaxp_idx = -1;

	double Gmaxn = -INF;
	double Gmaxn2 = -INF;
	int Gmaxn_idx = -1;

	int Gmin_idx = -1;
	double obj_diff_min = INF;

	for(int t=0;t<active_size;t++)
		if(y[t]==+1)
		{
			if(!is_upper_bound(t))
				if(-G[t] >= Gmaxp)
				{
					Gmaxp = -G[t];
					Gmaxp_idx = t;
				}
		}
		else
		{
			if(!is_lower_bound(t))
				if(G[t] >= Gmaxn)
				{
					Gmaxn = G[t];
					Gmaxn_idx = t;
				}
		}

	int ip = Gmaxp_idx;
	int in = Gmaxn_idx;
	const Qfloat *Q_ip = NULL;
	const Qfloat *Q_in = NULL;
	if(ip != -1) // NULL Q_ip not accessed: Gmaxp=-INF if ip=-1
		Q_ip = Q->get_Q(ip,active_size);
	if(in != -1)
		Q_in = Q->get_Q(in,active_size);

	for(int j=0;j<active_size;j++)
	{
		if(y[j]==+1)
		{
			if (!is_lower_bound(j))	
			{
				double grad_diff=Gmaxp+G[j];
				if (G[j] >= Gmaxp2)
					Gmaxp2 = G[j];
				if (grad_diff > 0)
				{
					double obj_diff;
					double quad_coef = QD[ip]+QD[j]-2*Q_ip[j];
					if (quad_coef > 0)
						obj_diff = -(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff = -(grad_diff*grad_diff)/TAU;

					if (obj_diff <= obj_diff_min)
					{
						Gmin_idx=j;
						obj_diff_min = obj_diff;
					}
				}
			}
		}
		else
		{
			if (!is_upper_bound(j))
			{
				double grad_diff=Gmaxn-G[j];
				if (-G[j] >= Gmaxn2)
					Gmaxn2 = -G[j];
				if (grad_diff > 0)
				{
					double obj_diff;
					double quad_coef = QD[in]+QD[j]-2*Q_in[j];
					if (quad_coef > 0)
						obj_diff = -(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff = -(grad_diff*grad_diff)/TAU;

					if (obj_diff <= obj_diff_min)
					{
						Gmin_idx=j;
						obj_diff_min = obj_diff;
					}
				}
			}
		}
	}

	if(max(Gmaxp+Gmaxp2,Gmaxn+Gmaxn2) < eps)
		return 1;

	if (y[Gmin_idx] == +1)
		out_i = Gmaxp_idx;
	else
		out_i = Gmaxn_idx;
	out_j = Gmin_idx;

	return 0;
}

bool Solver_NU::be_shrunk(int i, double Gmax1, double Gmax2, double Gmax3, double Gmax4)
{
	if(is_upper_bound(i))
	{
		if(y[i]==+1)
			return(-G[i] > Gmax1);
		else	
			return(-G[i] > Gmax4);
	}
	else if(is_lower_bound(i))
	{
		if(y[i]==+1)
			return(G[i] > Gmax2);
		else	
			return(G[i] > Gmax3);
	}
	else
		return(false);
}

void Solver_NU::do_shrinking()
{
	double Gmax1 = -INF;	// max { -y_i * grad(f)_i | y_i = +1, i in I_up(\alpha) }
	double Gmax2 = -INF;	// max { y_i * grad(f)_i | y_i = +1, i in I_low(\alpha) }
	double Gmax3 = -INF;	// max { -y_i * grad(f)_i | y_i = -1, i in I_up(\alpha) }
	double Gmax4 = -INF;	// max { y_i * grad(f)_i | y_i = -1, i in I_low(\alpha) }

	// find maximal violating pair first
	int i;
	for(i=0;i<active_size;i++)
	{
		if(!is_upper_bound(i))
		{
			if(y[i]==+1)
			{
				if(-G[i] > Gmax1) Gmax1 = -G[i];
			}
			else	if(-G[i] > Gmax4) Gmax4 = -G[i];
		}
		if(!is_lower_bound(i))
		{
			if(y[i]==+1)
			{	
				if(G[i] > Gmax2) Gmax2 = G[i];
			}
			else	if(G[i] > Gmax3) Gmax3 = G[i];
		}
	}

	if(unshrink == false && max(Gmax1+Gmax2,Gmax3+Gmax4) <= eps*10) 
	{
		unshrink = true;
		reconstruct_gradient();
		active_size = l;
	}

	for(i=0;i<active_size;i++)
		if (be_shrunk(i, Gmax1, Gmax2, Gmax3, Gmax4))
		{
			active_size--;
			while (active_size > i)
			{
				if (!be_shrunk(active_size, Gmax1, Gmax2, Gmax3, Gmax4))
				{
					swap_index(i,active_size);
					break;
				}
				active_size--;
			}
		}
}

double Solver_NU::calculate_rho()
{
	int nr_free1 = 0,nr_free2 = 0;
	double ub1 = INF, ub2 = INF;
	double lb1 = -INF, lb2 = -INF;
	double sum_free1 = 0, sum_free2 = 0;

	for(int i=0;i<active_size;i++)
	{
		if(y[i]==+1)
		{
			if(is_upper_bound(i))
				lb1 = max(lb1,G[i]);
			else if(is_lower_bound(i))
				ub1 = min(ub1,G[i]);
			else
			{
				++nr_free1;
				sum_free1 += G[i];
			}
		}
		else
		{
			if(is_upper_bound(i))
				lb2 = max(lb2,G[i]);
			else if(is_lower_bound(i))
				ub2 = min(ub2,G[i]);
			else
			{
				++nr_free2;
				sum_free2 += G[i];
			}
		}
	}

	double r1,r2;
	if(nr_free1 > 0)
		r1 = sum_free1/nr_free1;
	else
		r1 = (ub1+lb1)/2;
	
	if(nr_free2 > 0)
		r2 = sum_free2/nr_free2;
	else
		r2 = (ub2+lb2)/2;
	
	si->r = (r1+r2)/2;
	return (r1-r2)/2;
}

//
// Q matrices for various formulations
//
class SVC_Q: public Kernel
{ 
public:
	SVC_Q(const svm_problem& prob, const svm_parameter& param, const schar *y_)
	:Kernel(prob.l, prob.x, param)
	{
		clone(y,y_,prob.l);
		cache = new Cache(prob.l, param.cache_size);  //控制cache的大小
		QD = new double[prob.l];
		for(int i=0;i<prob.l;i++)
			QD[i] = (this->*kernel_function)(i,i);
	}

    bool is_cached(int i, int len) const {
		return cache->is_cached(i, len);
	}

	Qfloat get_Q_ij(int i, int j) const {
		if(!useShrinkAndCache)
		    return (Qfloat)(y[i] * y[j] * (this->*kernel_function)(i, j));
		else {
			Qfloat qf = cache->get_Q_ij(i, j);
			if(qf == NULL)
				qf = (Qfloat)(y[i] * y[j] * (this->*kernel_function)(i, j));
			return qf;
		}
	}
	Qfloat *get_Q(int i, int len) const
	{   
		//non-cache version
		Qfloat *data;
		{
			data= new Qfloat[len]; //不用cache，这样由于没有cache,在调用完后应用delete[].
			for(int j=0;j<len;j++)
			{
				data[j] = (Qfloat)(y[i]*y[j]*(this->*kernel_function)(i,j));
			}
		}
		return data;
	}
	Qfloat *get_Q(int i, int len,int flag) const //只为了统计cache的命中次数，且这里不提供non-cache版本
	{
		Qfloat *data;
		int start, j;
		if((start = cache->get_data(i,&data,len,flag)) < len)
		{
			for(j=start;j<len;j++)
				data[j] = (Qfloat)(y[i]*y[j]*(this->*kernel_function)(i,j));
		}

		return data;
	}
	void getHitInfo(long&totCnts,long& hitCnts,long& hitEls) const
	{
		cache->getHitInfo(totCnts,hitCnts,hitEls);
	}
	double *get_QD() const
	{
		return QD;
	}

	void swap_index(int i, int j) const
	{
		cache->swap_index(i,j);
		Kernel::swap_index(i,j);
		swap(y[i],y[j]);
		swap(QD[i],QD[j]);
	}

	~SVC_Q()
	{
		delete[] y;
		delete cache;
		delete[] QD;
	}
private:
	schar *y;
	Cache *cache;
	double *QD;
};

class ONE_CLASS_Q: public Kernel
{
public:
	ONE_CLASS_Q(const svm_problem& prob, const svm_parameter& param)
	:Kernel(prob.l, prob.x, param)
	{
		cache = new Cache(prob.l,(long int)(param.cache_size*(1<<20)));
		QD = new double[prob.l];
		for(int i=0;i<prob.l;i++)
			QD[i] = (this->*kernel_function)(i,i);
	}
	
	Qfloat *get_Q(int i, int len) const
	{
		Qfloat *data;
		int start, j;
		if((start = cache->get_data(i,&data,len)) < len)
		{
			for(j=start;j<len;j++)
				data[j] = (Qfloat)(this->*kernel_function)(i,j);
		}
		return data;
	}

	double *get_QD() const
	{
		return QD;
	}

	void swap_index(int i, int j) const
	{
		cache->swap_index(i,j);
		Kernel::swap_index(i,j);
		swap(QD[i],QD[j]);
	}

	~ONE_CLASS_Q()
	{
		delete cache;
		delete[] QD;
	}
private:
	Cache *cache;
	double *QD;
};

class SVR_Q: public Kernel
{ 
public:
	SVR_Q(const svm_problem& prob, const svm_parameter& param)
	:Kernel(prob.l, prob.x, param)
	{
		l = prob.l;
		cache = new Cache(l,(long int)(param.cache_size*(1<<20)));
		QD = new double[2*l];
		sign = new schar[2*l];
		index = new int[2*l];
		for(int k=0;k<l;k++)
		{
			sign[k] = 1;
			sign[k+l] = -1;
			index[k] = k;
			index[k+l] = k;
			QD[k] = (this->*kernel_function)(k,k);
			QD[k+l] = QD[k];
		}
		buffer[0] = new Qfloat[2*l];
		buffer[1] = new Qfloat[2*l];
		next_buffer = 0;
	}

	void swap_index(int i, int j) const
	{
		swap(sign[i],sign[j]);
		swap(index[i],index[j]);
		swap(QD[i],QD[j]);
	}
	
	Qfloat *get_Q(int i, int len) const
	{
		Qfloat *data;
		int j, real_i = index[i];
		if(cache->get_data(real_i,&data,l) < l)
		{
			for(j=0;j<l;j++)
				data[j] = (Qfloat)(this->*kernel_function)(real_i,j);
		}

		// reorder and copy
		Qfloat *buf = buffer[next_buffer];
		next_buffer = 1 - next_buffer;
		schar si = sign[i];
		for(j=0;j<len;j++)
			buf[j] = (Qfloat) si * (Qfloat) sign[j] * data[index[j]];
		return buf;
	}

	double *get_QD() const
	{
		return QD;
	}

	~SVR_Q()
	{
		delete cache;
		delete[] sign;
		delete[] index;
		delete[] buffer[0];
		delete[] buffer[1];
		delete[] QD;
	}
private:
	int l;
	Cache *cache;
	schar *sign;
	int *index;
	mutable int next_buffer;
	Qfloat *buffer[2];
	double *QD;
};

//
// construct and solve various formulations
//
static void solve_c_svc(
	const svm_problem *prob, const svm_parameter* param,
	double *alpha, Solver::SolutionInfo* si, double Cp, double Cn)
{
	int l = prob->l;
	double *minus_ones = new double[l];
	schar *y = new schar[l];

	int i;

	for(i=0;i<l;i++)
	{
		alpha[i] = 0;
		minus_ones[i] = -1;
		if(prob->y[i] > 0) y[i] = +1; else y[i] = -1;
	}
	Solver s;

	LARGE_INTEGER lp; 
    double dfFreq; 
    QueryPerformanceFrequency(&lp);
    dfFreq = (double)lp.QuadPart;// 获得计数器的时钟频率 
	QueryPerformanceCounter(&lp);
	LONGLONG dwStart=lp.QuadPart;

	if (parallelSize == 1) {/*单路*/
		if (useShrinkAndCache) //使用优化技术
			s.Solve(l, SVC_Q(*prob, *param, y), minus_ones, y,
				alpha, Cp, Cn, param->eps, si, param->shrinking);
		else //不使用优化技术
			s.Solve2(l, SVC_Q(*prob, *param, y), minus_ones, y,
				alpha, Cp, Cn, param->eps, si);
	}
	else {/*多路*/
		s.Solve3(l, SVC_Q(*prob, *param, y), minus_ones, y,
			alpha, Cp, Cn, param->eps, si);
	}
	
	QueryPerformanceCounter(&lp);
	LONGLONG dwEnd=lp.QuadPart;
	double Sub_Solver = (double)(dwEnd - dwStart)/dfFreq;

	//cout<<"Sub-Solver Time: " <<Sub_Solver<<"s"<<endl;
	info("Sub-Solver Time: %fs\n", Sub_Solver);
	fprintf(fout,"Sub-Solver Time: %fs\n",Sub_Solver);
	total_solver_time+=Sub_Solver; // sum up solver time
	double sum_alpha=0;
	for(i=0;i<l;i++)
		sum_alpha += alpha[i];

	if (Cp==Cn)
	{
		info("nu = %f\n", sum_alpha/(Cp*prob->l));
		fprintf(fout,"nu = %f\n", sum_alpha/(Cp*prob->l));
	}

	for(i=0;i<l;i++)
		alpha[i] *= y[i];

	delete[] minus_ones;
	delete[] y;
}

static void solve_nu_svc(
	const svm_problem *prob, const svm_parameter *param,
	double *alpha, Solver::SolutionInfo* si)
{
	int i;
	int l = prob->l;
	double nu = param->nu;

	schar *y = new schar[l];

	for(i=0;i<l;i++)
		if(prob->y[i]>0)
			y[i] = +1;
		else
			y[i] = -1;

	double sum_pos = nu*l/2;
	double sum_neg = nu*l/2;

	for(i=0;i<l;i++)
		if(y[i] == +1)
		{
			alpha[i] = min(1.0,sum_pos);
			sum_pos -= alpha[i];
		}
		else
		{
			alpha[i] = min(1.0,sum_neg);
			sum_neg -= alpha[i];
		}

	double *zeros = new double[l];

	for(i=0;i<l;i++)
		zeros[i] = 0;

	Solver_NU s;
	s.Solve(l, SVC_Q(*prob,*param,y), zeros, y,
		alpha, 1.0, 1.0, param->eps, si,  param->shrinking);
	double r = si->r;

	info("C = %f\n",1/r);

	for(i=0;i<l;i++)
		alpha[i] *= y[i]/r;

	si->rho /= r;
	si->obj /= (r*r);
	si->upper_bound_p = 1/r;
	si->upper_bound_n = 1/r;

	delete[] y;
	delete[] zeros;
}

static void solve_one_class(
	const svm_problem *prob, const svm_parameter *param,
	double *alpha, Solver::SolutionInfo* si)
{
	int l = prob->l;
	double *zeros = new double[l];
	schar *ones = new schar[l];
	int i;

	int n = (int)(param->nu*prob->l);	// # of alpha's at upper bound

	for(i=0;i<n;i++)
		alpha[i] = 1;
	if(n<prob->l)
		alpha[n] = param->nu * prob->l - n;
	for(i=n+1;i<l;i++)
		alpha[i] = 0;

	for(i=0;i<l;i++)
	{
		zeros[i] = 0;
		ones[i] = 1;
	}

	Solver s;
	s.Solve(l, ONE_CLASS_Q(*prob,*param), zeros, ones,
		alpha, 1.0, 1.0, param->eps, si, param->shrinking);

	delete[] zeros;
	delete[] ones;
}

static void solve_epsilon_svr(
	const svm_problem *prob, const svm_parameter *param,
	double *alpha, Solver::SolutionInfo* si)
{
	int l = prob->l;
	double *alpha2 = new double[2*l];
	double *linear_term = new double[2*l];
	schar *y = new schar[2*l];
	int i;

	for(i=0;i<l;i++)
	{
		alpha2[i] = 0;
		linear_term[i] = param->p - prob->y[i];
		y[i] = 1;

		alpha2[i+l] = 0;
		linear_term[i+l] = param->p + prob->y[i];
		y[i+l] = -1;
	}

	Solver s;
	s.Solve(2*l, SVR_Q(*prob,*param), linear_term, y,
		alpha2, param->C, param->C, param->eps, si, param->shrinking);

	double sum_alpha = 0;
	for(i=0;i<l;i++)
	{
		alpha[i] = alpha2[i] - alpha2[i+l];
		sum_alpha += fabs(alpha[i]);
	}
	info("nu = %f\n",sum_alpha/(param->C*l));

	delete[] alpha2;
	delete[] linear_term;
	delete[] y;
}

static void solve_nu_svr(
	const svm_problem *prob, const svm_parameter *param,
	double *alpha, Solver::SolutionInfo* si)
{
	int l = prob->l;
	double C = param->C;
	double *alpha2 = new double[2*l];
	double *linear_term = new double[2*l];
	schar *y = new schar[2*l];
	int i;

	double sum = C * param->nu * l / 2;
	for(i=0;i<l;i++)
	{
		alpha2[i] = alpha2[i+l] = min(sum,C);
		sum -= alpha2[i];

		linear_term[i] = - prob->y[i];
		y[i] = 1;

		linear_term[i+l] = prob->y[i];
		y[i+l] = -1;
	}

	Solver_NU s;
	s.Solve(2*l, SVR_Q(*prob,*param), linear_term, y,
		alpha2, C, C, param->eps, si, param->shrinking);

	info("epsilon = %f\n",-si->r);

	for(i=0;i<l;i++)
		alpha[i] = alpha2[i] - alpha2[i+l];

	delete[] alpha2;
	delete[] linear_term;
	delete[] y;
}

//
// decision_function
//
struct decision_function
{
	double *alpha;
	double rho;
};

static decision_function svm_train_one(
	const svm_problem *prob, const svm_parameter *param,
	double Cp, double Cn)
{
	double *alpha = Malloc(double,prob->l);
	Solver::SolutionInfo si;
	switch(param->svm_type)
	{
		case C_SVC:
			solve_c_svc(prob,param,alpha,&si,Cp,Cn);
			break;
		case NU_SVC:
			solve_nu_svc(prob,param,alpha,&si);
			break;
		case ONE_CLASS:
			solve_one_class(prob,param,alpha,&si);
			break;
		case EPSILON_SVR:
			solve_epsilon_svr(prob,param,alpha,&si);
			break;
		case NU_SVR:
			solve_nu_svr(prob,param,alpha,&si);
			break;
	}

	info("obj = %f, rho = %f\n",si.obj,si.rho);
    fprintf(fout,"obj = %f, rho = %f\n",si.obj,si.rho);
	// output SVs

	int nSV = 0;
	int nBSV = 0;
	for(int i=0;i<prob->l;i++)
	{
		if(fabs(alpha[i]) > 0)
		{
			++nSV;
			if(prob->y[i] > 0)
			{
				if(fabs(alpha[i]) >= si.upper_bound_p)
					++nBSV;
			}
			else
			{
				if(fabs(alpha[i]) >= si.upper_bound_n)
					++nBSV;
			}
		}
	}

	info("nSV = %d, nBSV = %d\n",nSV,nBSV);
	fprintf(fout,"nSV = %d, nBSV = %d\n",nSV,nBSV);

	decision_function f;
	f.alpha = alpha;
	f.rho = si.rho;
	return f;
}

// Platt's binary SVM Probablistic Output: an improvement from Lin et al.
static void sigmoid_train(
	int l, const double *dec_values, const double *labels, 
	double& A, double& B)
{
	double prior1=0, prior0 = 0;
	int i;

	for (i=0;i<l;i++)
		if (labels[i] > 0) prior1+=1;
		else prior0+=1;
	
	int max_iter=100;	// Maximal number of iterations
	double min_step=1e-10;	// Minimal step taken in line search
	double sigma=1e-12;	// For numerically strict PD of Hessian
	double eps=1e-5;
	double hiTarget=(prior1+1.0)/(prior1+2.0);
	double loTarget=1/(prior0+2.0);
	double *t=Malloc(double,l);
	double fApB,p,q,h11,h22,h21,g1,g2,det,dA,dB,gd,stepsize;
	double newA,newB,newf,d1,d2;
	int iter;
	
	// Initial Point and Initial Fun Value
	A=0.0; B=log((prior0+1.0)/(prior1+1.0));
	double fval = 0.0;

	for (i=0;i<l;i++)
	{
		if (labels[i]>0) t[i]=hiTarget;
		else t[i]=loTarget;
		fApB = dec_values[i]*A+B;
		if (fApB>=0)
			fval += t[i]*fApB + log(1+exp(-fApB));
		else
			fval += (t[i] - 1)*fApB +log(1+exp(fApB));
	}
	for (iter=0;iter<max_iter;iter++)
	{
		// Update Gradient and Hessian (use H' = H + sigma I)
		h11=sigma; // numerically ensures strict PD
		h22=sigma;
		h21=0.0;g1=0.0;g2=0.0;
		for (i=0;i<l;i++)
		{
			fApB = dec_values[i]*A+B;
			if (fApB >= 0)
			{
				p=exp(-fApB)/(1.0+exp(-fApB));
				q=1.0/(1.0+exp(-fApB));
			}
			else
			{
				p=1.0/(1.0+exp(fApB));
				q=exp(fApB)/(1.0+exp(fApB));
			}
			d2=p*q;
			h11+=dec_values[i]*dec_values[i]*d2;
			h22+=d2;
			h21+=dec_values[i]*d2;
			d1=t[i]-p;
			g1+=dec_values[i]*d1;
			g2+=d1;
		}

		// Stopping Criteria
		if (fabs(g1)<eps && fabs(g2)<eps)
			break;

		// Finding Newton direction: -inv(H') * g
		det=h11*h22-h21*h21;
		dA=-(h22*g1 - h21 * g2) / det;
		dB=-(-h21*g1+ h11 * g2) / det;
		gd=g1*dA+g2*dB;


		stepsize = 1;		// Line Search
		while (stepsize >= min_step)
		{
			newA = A + stepsize * dA;
			newB = B + stepsize * dB;

			// New function value
			newf = 0.0;
			for (i=0;i<l;i++)
			{
				fApB = dec_values[i]*newA+newB;
				if (fApB >= 0)
					newf += t[i]*fApB + log(1+exp(-fApB));
				else
					newf += (t[i] - 1)*fApB +log(1+exp(fApB));
			}
			// Check sufficient decrease
			if (newf<fval+0.0001*stepsize*gd)
			{
				A=newA;B=newB;fval=newf;
				break;
			}
			else
				stepsize = stepsize / 2.0;
		}

		if (stepsize < min_step)
		{
			info("Line search fails in two-class probability estimates\n");
			break;
		}
	}

	if (iter>=max_iter)
		info("Reaching maximal iterations in two-class probability estimates\n");
	free(t);
}

static double sigmoid_predict(double decision_value, double A, double B)
{
	double fApB = decision_value*A+B;
	// 1-p used later; avoid catastrophic cancellation
	if (fApB >= 0)
		return exp(-fApB)/(1.0+exp(-fApB));
	else
		return 1.0/(1+exp(fApB)) ;
}

// Method 2 from the multiclass_prob paper by Wu, Lin, and Weng
static void multiclass_probability(int k, double **r, double *p)
{
	int t,j;
	int iter = 0, max_iter=max(100,k);
	double **Q=Malloc(double *,k);
	double *Qp=Malloc(double,k);
	double pQp, eps=0.005/k;
	
	for (t=0;t<k;t++)
	{
		p[t]=1.0/k;  // Valid if k = 1
		Q[t]=Malloc(double,k);
		Q[t][t]=0;
		for (j=0;j<t;j++)
		{
			Q[t][t]+=r[j][t]*r[j][t];
			Q[t][j]=Q[j][t];
		}
		for (j=t+1;j<k;j++)
		{
			Q[t][t]+=r[j][t]*r[j][t];
			Q[t][j]=-r[j][t]*r[t][j];
		}
	}
	for (iter=0;iter<max_iter;iter++)
	{
		// stopping condition, recalculate QP,pQP for numerical accuracy
		pQp=0;
		for (t=0;t<k;t++)
		{
			Qp[t]=0;
			for (j=0;j<k;j++)
				Qp[t]+=Q[t][j]*p[j];
			pQp+=p[t]*Qp[t];
		}
		double max_error=0;
		for (t=0;t<k;t++)
		{
			double error=fabs(Qp[t]-pQp);
			if (error>max_error)
				max_error=error;
		}
		if (max_error<eps) break;
		
		for (t=0;t<k;t++)
		{
			double diff=(-Qp[t]+pQp)/Q[t][t];
			p[t]+=diff;
			pQp=(pQp+diff*(diff*Q[t][t]+2*Qp[t]))/(1+diff)/(1+diff);
			for (j=0;j<k;j++)
			{
				Qp[j]=(Qp[j]+diff*Q[t][j])/(1+diff);
				p[j]/=(1+diff);
			}
		}
	}
	if (iter>=max_iter)
		info("Exceeds max_iter in multiclass_prob\n");
	for(t=0;t<k;t++) free(Q[t]);
	free(Q);
	free(Qp);
}

// Cross-validation decision values for probability estimates
static void svm_binary_svc_probability(
	const svm_problem *prob, const svm_parameter *param,
	double Cp, double Cn, double& probA, double& probB)
{
	int i;
	int nr_fold = 5;
	int *perm = Malloc(int,prob->l);
	double *dec_values = Malloc(double,prob->l);

	// random shuffle
	for(i=0;i<prob->l;i++) perm[i]=i;
	for(i=0;i<prob->l;i++)
	{
		int j = i+rand()%(prob->l-i);
		swap(perm[i],perm[j]);
	}
	for(i=0;i<nr_fold;i++)
	{
		int begin = i*prob->l/nr_fold;
		int end = (i+1)*prob->l/nr_fold;
		int j,k;
		struct svm_problem subprob;

		subprob.l = prob->l-(end-begin);
		subprob.x = Malloc(struct svm_node*,subprob.l);
		subprob.y = Malloc(double,subprob.l);
			
		k=0;
		for(j=0;j<begin;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		for(j=end;j<prob->l;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		int p_count=0,n_count=0;
		for(j=0;j<k;j++)
			if(subprob.y[j]>0)
				p_count++;
			else
				n_count++;

		if(p_count==0 && n_count==0)
			for(j=begin;j<end;j++)
				dec_values[perm[j]] = 0;
		else if(p_count > 0 && n_count == 0)
			for(j=begin;j<end;j++)
				dec_values[perm[j]] = 1;
		else if(p_count == 0 && n_count > 0)
			for(j=begin;j<end;j++)
				dec_values[perm[j]] = -1;
		else
		{
			svm_parameter subparam = *param;
			subparam.probability=0;
			subparam.C=1.0;
			subparam.nr_weight=2;
			subparam.weight_label = Malloc(int,2);
			subparam.weight = Malloc(double,2);
			subparam.weight_label[0]=+1;
			subparam.weight_label[1]=-1;
			subparam.weight[0]=Cp;
			subparam.weight[1]=Cn;
			struct svm_model *submodel = svm_train(&subprob,&subparam,NULL);
			for(j=begin;j<end;j++)
			{
				svm_predict_values(submodel,prob->x[perm[j]],&(dec_values[perm[j]]));
				// ensure +1 -1 order; reason not using CV subroutine
				dec_values[perm[j]] *= submodel->label[0];
			}		
			svm_free_and_destroy_model(&submodel);
			svm_destroy_param(&subparam);
		}
		free(subprob.x);
		free(subprob.y);
	}		
	sigmoid_train(prob->l,dec_values,prob->y,probA,probB);
	free(dec_values);
	free(perm);
}

// Return parameter of a Laplace distribution 
static double svm_svr_probability(
	const svm_problem *prob, const svm_parameter *param)
{
	int i;
	int nr_fold = 5;
	double *ymv = Malloc(double,prob->l);
	double mae = 0;

	svm_parameter newparam = *param;
	newparam.probability = 0;
	svm_cross_validation(prob,&newparam,nr_fold,ymv);
	for(i=0;i<prob->l;i++)
	{
		ymv[i]=prob->y[i]-ymv[i];
		mae += fabs(ymv[i]);
	}		
	mae /= prob->l;
	double std=sqrt(2*mae*mae);
	int count=0;
	mae=0;
	for(i=0;i<prob->l;i++)
		if (fabs(ymv[i]) > 5*std) 
			count=count+1;
		else 
			mae+=fabs(ymv[i]);
	mae /= (prob->l-count);
	info("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma= %g\n",mae);
	free(ymv);
	return mae;
}


// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length l, must be allocated before calling this subroutine
static void svm_group_classes(const svm_problem *prob, int *nr_class_ret, int **label_ret, int **start_ret, int **count_ret, int *perm)
{
	int l = prob->l;
	int max_nr_class = 16;
	int nr_class = 0;
	int *label = Malloc(int,max_nr_class);
	int *count = Malloc(int,max_nr_class);
	int *data_label = Malloc(int,l);
	int i;

	/*映射每个样本所属的label，label[nr_class]存放原样本集中的类别, data_label[l]存放每个样本的类别在label[]中的下标索引*/
	for(i=0;i<l;i++)
	{
		int this_label = (int)prob->y[i];
		int j;
		for(j=0;j<nr_class;j++)
		{
			if(this_label == label[j])
			{
				++count[j];
				break;
			}
		}
		data_label[i] = j;
		if(j == nr_class)
		{
			if(nr_class == max_nr_class)
			{
				max_nr_class *= 2;
				label = (int *)realloc(label,max_nr_class*sizeof(int));
				count = (int *)realloc(count,max_nr_class*sizeof(int));
			}
			label[nr_class] = this_label;
			count[nr_class] = 1;  /*统计每个样本类别的实际个数*/
			++nr_class;
		}
	}

	//
	// Labels are ordered by their first occurrence in the training set. 
	// However, for two-class sets with -1/+1 labels and -1 appears first, 
	// we swap labels to ensure that internally the binary SVM has positive data corresponding to the +1 instances.
	//
	if (nr_class == 2 && label[0] == -1 && label[1] == 1)
	{
		swap(label[0],label[1]);
		swap(count[0],count[1]);
		for(i=0;i<l;i++)
		{
			if(data_label[i] == 0)
				data_label[i] = 1;
			else
				data_label[i] = 0;
		}
	}

	int *start = Malloc(int,nr_class);
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];
	/*将同类别的样本编号聚集的方式放在perm[]中，start[i]标识类别label[i]在perm存放的起始索引, count[i]则为对应个数*/
	for(i=0;i<l;i++)
	{
		perm[start[data_label[i]]] = i;
		++start[data_label[i]];
	}
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];

	*nr_class_ret = nr_class;
	*label_ret = label;
	*start_ret = start;
	*count_ret = count;
	free(data_label);
}

//
// Interface functions
//
svm_model *svm_train(const svm_problem *prob, const svm_parameter *param,char *input_file_name)
{  

	{   /*准备好相应的输出文件*/
		output_result_file=new char[1024];
		strcpy(output_result_file,input_file_name); //copy input file name
		sprintf(output_result_file,"%s_output_result.txt",output_result_file);
		fout = fopen(output_result_file,"w");
		{//2014-10-30
			char output_obj_file[100];
			char output_flag_file[100];
			strcpy(output_obj_file,input_file_name); 
			sprintf(output_obj_file,"%s_obj.txt",output_obj_file);    /*_obj结尾的保存每次迭代过程中的*/
			strcpy(output_flag_file,input_file_name); 
			sprintf(output_flag_file,"%s_flag.txt",output_flag_file); /*_flag结尾的文件保存每次迭代的实际并行路数*/
			fout_obj=fopen(output_obj_file,"w");
			fout_flag=fopen(output_flag_file,"w");
		}
	
	}
	svm_model *model = Malloc(svm_model,1);
	model->param = *param;
	model->free_sv = 0;	// XXX

	if(param->svm_type == ONE_CLASS ||
	   param->svm_type == EPSILON_SVR ||
	   param->svm_type == NU_SVR)
	{
		// regression or one-class-svm
		model->nr_class = 2;
		model->label = NULL;
		model->nSV = NULL;
		model->probA = NULL; model->probB = NULL;
		model->sv_coef = Malloc(double *,1);

		if(param->probability && 
		   (param->svm_type == EPSILON_SVR ||
		    param->svm_type == NU_SVR))
		{
			model->probA = Malloc(double,1);
			model->probA[0] = svm_svr_probability(prob,param);
		}

		decision_function f = svm_train_one(prob,param,0,0);
		model->rho = Malloc(double,1);
		model->rho[0] = f.rho;

		int nSV = 0;
		int i;
		for(i=0;i<prob->l;i++)
			if(fabs(f.alpha[i]) > 0) ++nSV;
		model->l = nSV;
		model->SV = Malloc(svm_node *,nSV);
		model->sv_coef[0] = Malloc(double,nSV);
		model->sv_indices = Malloc(int,nSV);
		int j = 0;
		for(i=0;i<prob->l;i++)
			if(fabs(f.alpha[i]) > 0)
			{
				model->SV[j] = prob->x[i];
				model->sv_coef[0][j] = f.alpha[i];
				model->sv_indices[j] = i+1;
				++j;
			}		

		free(f.alpha);
	}
	else
	{
		// classification
		int l = prob->l;
		int nr_class;
		int *label = NULL;
		int *start = NULL;
		int *count = NULL;
		int *perm = Malloc(int,l);

		// group training data of the same class
		/*将同类别的属性排列在一起
		 nr_class: 类别个数
		 label[nr_class]: 将样本中的原类别标识放入label[]中
		 perm[l]: 属于同类别的样本索引排列在一起
		 start[nr_class]: 每一类别在perm中存放的起始索引
		 count[nr_class]: 每一类别在perm中的实际个数
		*/
		svm_group_classes(prob,&nr_class,&label,&start,&count,perm);
		if(nr_class == 1) 
			info("WARNING: training data in only one class. See README for details.\n");
		
		svm_node **x = Malloc(svm_node *,l);
		int i;
		for(i=0;i<l;i++)
			x[i] = prob->x[perm[i]]; //x[l]对应训练中用到的样本序列

		// calculate weighted C

		double *weighted_C = Malloc(double, nr_class);
		for(i=0;i<nr_class;i++)
			weighted_C[i] = param->C;
		for(i=0;i<param->nr_weight;i++)
		{	
			int j;
			for(j=0;j<nr_class;j++)
				if(param->weight_label[i] == label[j])
					break;
			if(j == nr_class)
				fprintf(stderr,"WARNING: class label %d specified in weight is not found\n", param->weight_label[i]);
			else
				weighted_C[j] *= param->weight[i];
		}

		// train k*(k-1)/2 models
		
		bool *nonzero = Malloc(bool,l);
		for(i=0;i<l;i++)
			nonzero[i] = false;
		decision_function *f = Malloc(decision_function,nr_class*(nr_class-1)/2);

		double *probA=NULL,*probB=NULL;
		if (param->probability)
		{
			probA=Malloc(double,nr_class*(nr_class-1)/2);
			probB=Malloc(double,nr_class*(nr_class-1)/2);
		}

		int p = 0;

		{   //每个数据集训练前进行初始化
		    totalVisits=0;
            totalCacheHits=0;
            totalCacheElements=0;

			multi_classTime=0; //训练总时间
			total_solver_time=0;//总时间
			total_iter=0;
		}
		fprintf(fout,"Now begin training dataset %s \n",input_file_name);
		for(i=0;i<nr_class;i++)
			for(int j=i+1;j<nr_class;j++)
			{
				svm_problem sub_prob;
				int si = start[i], sj = start[j];
				int ci = count[i], cj = count[j];
				sub_prob.l = ci+cj;
				sub_prob.x = Malloc(svm_node *,sub_prob.l);
				sub_prob.y = Malloc(double,sub_prob.l);
				int k;
				for(k=0;k<ci;k++)
				{
					sub_prob.x[k] = x[si+k];
					sub_prob.y[k] = +1;
				}
				for(k=0;k<cj;k++)
				{
					sub_prob.x[ci+k] = x[sj+k];
					sub_prob.y[ci+k] = -1;
				}

				if(param->probability)
					svm_binary_svc_probability(&sub_prob,param,weighted_C[i],weighted_C[j],probA[p],probB[p]);

				fprintf(fout, "<两两分类训练结果>");
				fprintf(fout,"\n%d's round: class %d vs. %d \n",p+1,label[i],label[j]);
				f[p] = svm_train_one(&sub_prob,param,weighted_C[i],weighted_C[j]); /*两两分类训练入口*/

				for(k=0;k<ci;k++)
					if(!nonzero[si+k] && fabs(f[p].alpha[k]) > 0)
						nonzero[si+k] = true;
				for(k=0;k<cj;k++)
					if(!nonzero[sj+k] && fabs(f[p].alpha[ci+k]) > 0)
						nonzero[sj+k] = true;
				free(sub_prob.x);
				free(sub_prob.y);
				++p;
			}

		// build output

		model->nr_class = nr_class;
		
		model->label = Malloc(int,nr_class);
		for(i=0;i<nr_class;i++)
			model->label[i] = label[i];
		
		model->rho = Malloc(double,nr_class*(nr_class-1)/2);
		for(i=0;i<nr_class*(nr_class-1)/2;i++)
			model->rho[i] = f[i].rho;

		if(param->probability)
		{
			model->probA = Malloc(double,nr_class*(nr_class-1)/2);
			model->probB = Malloc(double,nr_class*(nr_class-1)/2);
			for(i=0;i<nr_class*(nr_class-1)/2;i++)
			{
				model->probA[i] = probA[i];
				model->probB[i] = probB[i];
			}
		}
		else
		{
			model->probA=NULL;
			model->probB=NULL;
		}

		int total_sv = 0;
		int *nz_count = Malloc(int,nr_class);
		model->nSV = Malloc(int,nr_class);
		//统计每一个类别中的样本中包含多少支持向量
		for(i=0;i<nr_class;i++)
		{
			int nSV = 0;
			for(int j=0;j<count[i];j++)
				if(nonzero[start[i]+j])
				{	
					++nSV;
					++total_sv;
				}
			model->nSV[i] = nSV;
			nz_count[i] = nSV;
		}
		
		info("\n\nTotal nSV = %d\n",total_sv);
		info("Total #inters = %d\n",total_iter);

		info("Total # cache visits= %ld\n",totalVisits);
		info("Total # cache hits = %ld\n",totalCacheHits);
		info("Total # cache elements =%ld\n",totalCacheElements);

		model->l = total_sv; //支持向量集
		model->SV = Malloc(svm_node *,total_sv);
		model->sv_indices = Malloc(int,total_sv);
		p = 0;
		for(i=0;i<l;i++)
			if(nonzero[i])
			{
				model->SV[p] = x[i];
				model->sv_indices[p++] = perm[i] + 1; //映射到原样本中的索引
			}

		int *nz_start = Malloc(int,nr_class);
		nz_start[0] = 0;
		for(i=1;i<nr_class;i++)
			nz_start[i] = nz_start[i-1]+nz_count[i-1];

		model->sv_coef = Malloc(double *,nr_class-1); //注意: 是nr_class - 1, 而不是nr_class!
		for(i=0;i<nr_class-1;i++)
			model->sv_coef[i] = Malloc(double,total_sv);

		//对classifier(i)， 将与j两两训练过程中的alpha均存放到sv_coef[i][]中，数组sv_coef[i]的长度等于支持向量集的大小
		p = 0;
		for(i=0;i<nr_class;i++)
			for(int j=i+1;j<nr_class;j++)
			{
				// classifier (i,j): coefficients with
				// i are in sv_coef[j-1][nz_start[i]...],
				// j are in sv_coef[i][nz_start[j]...]

				int si = start[i];
				int sj = start[j];
				int ci = count[i];
				int cj = count[j];
				
				int q = nz_start[i];
				int k;
				for(k=0;k<ci;k++)
					if(nonzero[si+k])
						model->sv_coef[j-1][q++] = f[p].alpha[k];
				q = nz_start[j];
				for(k=0;k<cj;k++)
					if(nonzero[sj+k])
						model->sv_coef[i][q++] = f[p].alpha[ci+k];
				++p;
			}

		info("Time for multi-classification is %f\n ",multi_classTime);
		printf("Total Solver time is %fs\n",total_solver_time);

		fprintf(fout,"\n");
		fprintf(fout,"dataset: %s\n", input_file_name);

		fprintf(fout, "<训练参数>\n");
		fprintf(fout, "param.svm_type: %d\n", param->svm_type);
		fprintf(fout, "param.kernel_type: %d\n", param->kernel_type);
		fprintf(fout, "param.degree: %d\n", param->degree);
		fprintf(fout, "param.gamma: %.15f\n", param->gamma);
		fprintf(fout, "param.coef0: %f\n", param->coef0);
		fprintf(fout, "param.C: %f\n", param->C);
		fprintf(fout, "param.eps: %f\n", param->eps);
		fprintf(fout, "param.nr_weight: %d\n", param->nr_weight);
		fprintf(fout, "param.nu: %f\n", param->nu);
		fprintf(fout, "param.p: %f\n", param->p);
		fprintf(fout, "param.shrinking: %d\n", param->shrinking);
		fprintf(fout, "param.probability: %d\n", param->probability);
		fprintf(fout, "param.cache_size: %d\n", (useShrinkAndCache > 0? param->cache_size: 0));
		fprintf(fout, "param.parallels: %d\n",  parallelSize);
		fprintf(fout, "param.ilow_candidate_size: %d\n", parallelSize > 1? ilowCandidateSize: 0);
		fprintf(fout, "param.wss_algorithm(0:libsvm, 1:wss35): %s\n", useWSS35);
		fprintf(fout, "param.parallel_algorithm(0:IIPSMO, 1:IPSMO, 2:PSMO): %d\n", parallel_algorithm);
		fprintf(fout, "param.kernel_type(0:rbf, 1:linear, 2:poly, 3:sigmoid): %d\n", param->kernel_type);

		fprintf(fout, "<训练结果>\n");
		fprintf(fout,"Total nSV = %d\n",total_sv);
		fprintf(fout,"Total #inters = %d\n",total_iter);

		fprintf(fout,"Total # cache visits= %ld\n",totalVisits);
		fprintf(fout,"Total # cache hits = %ld\n",totalCacheHits);
		fprintf(fout,"Total # cache elements = %ld\n",totalCacheElements);

        fprintf(fout,"Total Multi-classification time is %fs\n",multi_classTime);
		fprintf(fout,"Total Solver time is %fs",total_solver_time);

	    model->totalIters=total_iter; 
		model->pureTime=multi_classTime;
		model->totalTime=total_solver_time;


		fclose(fout);
		fclose(fout_obj);
		fclose(fout_flag);
		free(output_result_file);
		
		free(label);
		free(probA);
		free(probB);
		free(count);
		free(perm);
		free(start);
		free(x);
		free(weighted_C);
		free(nonzero);
		for(i=0;i<nr_class*(nr_class-1)/2;i++)
			free(f[i].alpha);
		free(f);
		free(nz_count);
		free(nz_start);
	}
	return model;
}

// Stratified cross validation
void svm_cross_validation(const svm_problem *prob, const svm_parameter *param, int nr_fold, double *target)
{
	int i;
	int *fold_start;
	int l = prob->l;
	int *perm = Malloc(int,l);
	int nr_class;
	if (nr_fold > l)
	{
		nr_fold = l;
		fprintf(stderr,"WARNING: # folds > # data. Will use # folds = # data instead (i.e., leave-one-out cross validation)\n");
	}
	fold_start = Malloc(int,nr_fold+1);
	// stratified cv may not give leave-one-out rate
	// Each class to l folds -> some folds may have zero elements
	if((param->svm_type == C_SVC ||
	    param->svm_type == NU_SVC) && nr_fold < l)
	{
		int *start = NULL;
		int *label = NULL;
		int *count = NULL;
		svm_group_classes(prob,&nr_class,&label,&start,&count,perm);

		// random shuffle and then data grouped by fold using the array perm
		int *fold_count = Malloc(int,nr_fold);
		int c;
		int *index = Malloc(int,l);
		for(i=0;i<l;i++)
			index[i]=perm[i];
		for (c=0; c<nr_class; c++) 
			for(i=0;i<count[c];i++)
			{
				int j = i+rand()%(count[c]-i);
				swap(index[start[c]+j],index[start[c]+i]);
			}
		for(i=0;i<nr_fold;i++)
		{
			fold_count[i] = 0;
			for (c=0; c<nr_class;c++)
				fold_count[i]+=(i+1)*count[c]/nr_fold-i*count[c]/nr_fold;
		}
		fold_start[0]=0;
		for (i=1;i<=nr_fold;i++)
			fold_start[i] = fold_start[i-1]+fold_count[i-1];
		for (c=0; c<nr_class;c++)
			for(i=0;i<nr_fold;i++)
			{
				int begin = start[c]+i*count[c]/nr_fold;
				int end = start[c]+(i+1)*count[c]/nr_fold;
				for(int j=begin;j<end;j++)
				{
					perm[fold_start[i]] = index[j];
					fold_start[i]++;
				}
			}
		fold_start[0]=0;
		for (i=1;i<=nr_fold;i++)
			fold_start[i] = fold_start[i-1]+fold_count[i-1];
		free(start);
		free(label);
		free(count);
		free(index);
		free(fold_count);
	}
	else
	{
		for(i=0;i<l;i++) perm[i]=i;
		for(i=0;i<l;i++)
		{
			int j = i+rand()%(l-i);
			swap(perm[i],perm[j]);
		}
		for(i=0;i<=nr_fold;i++)
			fold_start[i]=i*l/nr_fold;
	}

	for(i=0;i<nr_fold;i++)
	{
		int begin = fold_start[i];
		int end = fold_start[i+1];
		int j,k;
		struct svm_problem subprob;

		subprob.l = l-(end-begin);
		subprob.x = Malloc(struct svm_node*,subprob.l);
		subprob.y = Malloc(double,subprob.l);
			
		k=0;
		for(j=0;j<begin;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		for(j=end;j<l;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		struct svm_model *submodel = svm_train(&subprob,param,NULL);
		if(param->probability && 
		   (param->svm_type == C_SVC || param->svm_type == NU_SVC))
		{
			double *prob_estimates=Malloc(double,svm_get_nr_class(submodel));
			for(j=begin;j<end;j++)
				target[perm[j]] = svm_predict_probability(submodel,prob->x[perm[j]],prob_estimates);
			free(prob_estimates);
		}
		else
			for(j=begin;j<end;j++)
				target[perm[j]] = svm_predict(submodel,prob->x[perm[j]]);
		svm_free_and_destroy_model(&submodel);
		free(subprob.x);
		free(subprob.y);
	}		
	free(fold_start);
	free(perm);
}


int svm_get_svm_type(const svm_model *model)
{
	return model->param.svm_type;
}

int svm_get_nr_class(const svm_model *model)
{
	return model->nr_class;
}

void svm_get_labels(const svm_model *model, int* label)
{
	if (model->label != NULL)
		for(int i=0;i<model->nr_class;i++)
			label[i] = model->label[i];
}

void svm_get_sv_indices(const svm_model *model, int* indices)
{
	if (model->sv_indices != NULL)
		for(int i=0;i<model->l;i++)
			indices[i] = model->sv_indices[i];
}

int svm_get_nr_sv(const svm_model *model)
{
	return model->l;
}

double svm_get_svr_probability(const svm_model *model)
{
	if ((model->param.svm_type == EPSILON_SVR || model->param.svm_type == NU_SVR) &&
	    model->probA!=NULL)
		return model->probA[0];
	else
	{
		fprintf(stderr,"Model doesn't contain information for SVR probability inference\n");
		return 0;
	}
}

double svm_predict_values(const svm_model *model, const svm_node *x, double* dec_values)
{
	int i;
	if(model->param.svm_type == ONE_CLASS ||
	   model->param.svm_type == EPSILON_SVR ||
	   model->param.svm_type == NU_SVR)
	{
		double *sv_coef = model->sv_coef[0];
		double sum = 0;
		for(i=0;i<model->l;i++)
			sum += sv_coef[i] * Kernel::k_function(x,model->SV[i],model->param);
		sum -= model->rho[0];
		*dec_values = sum;

		if(model->param.svm_type == ONE_CLASS)
			return (sum>0)?1:-1;
		else
			return sum;
	}
	else
	{
		int nr_class = model->nr_class;
		int l = model->l;
		
		double *kvalue = Malloc(double,l);
		for(i=0;i<l;i++)
			kvalue[i] = Kernel::k_function(x,model->SV[i],model->param);

		int *start = Malloc(int,nr_class);
		start[0] = 0;
		for(i=1;i<nr_class;i++)
			start[i] = start[i-1]+model->nSV[i-1];

		int *vote = Malloc(int,nr_class);
		for(i=0;i<nr_class;i++)
			vote[i] = 0;

		int p=0;
		for(i=0;i<nr_class;i++)
			for(int j=i+1;j<nr_class;j++)
			{
				double sum = 0;
				int si = start[i];
				int sj = start[j];
				int ci = model->nSV[i];
				int cj = model->nSV[j];
				
				int k;
				double *coef1 = model->sv_coef[j-1];
				double *coef2 = model->sv_coef[i];
				for(k=0;k<ci;k++)
					sum += coef1[si+k] * kvalue[si+k];
				for(k=0;k<cj;k++)
					sum += coef2[sj+k] * kvalue[sj+k];
				sum -= model->rho[p];
				dec_values[p] = sum;

				if(dec_values[p] > 0)
					++vote[i];
				else
					++vote[j];
				p++;
			}

		int vote_max_idx = 0;
		for(i=1;i<nr_class;i++)
			if(vote[i] > vote[vote_max_idx])
				vote_max_idx = i;

		free(kvalue);
		free(start);
		free(vote);
		return model->label[vote_max_idx];
	}
}

double svm_predict(const svm_model *model, const svm_node *x)
{
	int nr_class = model->nr_class;
	double *dec_values;
	if(model->param.svm_type == ONE_CLASS ||
	   model->param.svm_type == EPSILON_SVR ||
	   model->param.svm_type == NU_SVR)
		dec_values = Malloc(double, 1);
	else 
		dec_values = Malloc(double, nr_class*(nr_class-1)/2);
	double pred_result = svm_predict_values(model, x, dec_values);
	free(dec_values);
	return pred_result;
}

double svm_predict_probability(
	const svm_model *model, const svm_node *x, double *prob_estimates)
{
	if ((model->param.svm_type == C_SVC || model->param.svm_type == NU_SVC) &&
	    model->probA!=NULL && model->probB!=NULL)
	{
		int i;
		int nr_class = model->nr_class;
		double *dec_values = Malloc(double, nr_class*(nr_class-1)/2);
		svm_predict_values(model, x, dec_values);

		double min_prob=1e-7;
		double **pairwise_prob=Malloc(double *,nr_class);
		for(i=0;i<nr_class;i++)
			pairwise_prob[i]=Malloc(double,nr_class);
		int k=0;
		for(i=0;i<nr_class;i++)
			for(int j=i+1;j<nr_class;j++)
			{
				pairwise_prob[i][j]=min(max(sigmoid_predict(dec_values[k],model->probA[k],model->probB[k]),min_prob),1-min_prob);
				pairwise_prob[j][i]=1-pairwise_prob[i][j];
				k++;
			}
		multiclass_probability(nr_class,pairwise_prob,prob_estimates);

		int prob_max_idx = 0;
		for(i=1;i<nr_class;i++)
			if(prob_estimates[i] > prob_estimates[prob_max_idx])
				prob_max_idx = i;
		for(i=0;i<nr_class;i++)
			free(pairwise_prob[i]);
		free(dec_values);
		free(pairwise_prob);
		return model->label[prob_max_idx];
	}
	else 
		return svm_predict(model, x);
}

static const char *svm_type_table[] =
{
	"c_svc","nu_svc","one_class","epsilon_svr","nu_svr",NULL
};

static const char *kernel_type_table[]=
{
	"linear","polynomial","rbf","sigmoid","precomputed",NULL
};

int svm_save_model(const char *model_file_name, const svm_model *model)
{
	FILE *fp = fopen(model_file_name,"w");
	if(fp==NULL) return -1;

	char *old_locale = strdup(setlocale(LC_ALL, NULL));
	setlocale(LC_ALL, "C");

	const svm_parameter& param = model->param;

	fprintf(fp,"svm_type %s\n", svm_type_table[param.svm_type]);
	fprintf(fp,"kernel_type %s\n", kernel_type_table[param.kernel_type]);

	if(param.kernel_type == POLY)
		fprintf(fp,"degree %d\n", param.degree);

	if(param.kernel_type == POLY || param.kernel_type == RBF || param.kernel_type == SIGMOID)
		fprintf(fp,"gamma %g\n", param.gamma);

	if(param.kernel_type == POLY || param.kernel_type == SIGMOID)
		fprintf(fp,"coef0 %g\n", param.coef0);

	int nr_class = model->nr_class;
	int l = model->l;
	fprintf(fp, "nr_class %d\n", nr_class);
	fprintf(fp, "total_sv %d\n",l);
	
	{
		fprintf(fp, "rho");
		for(int i=0;i<nr_class*(nr_class-1)/2;i++)
			fprintf(fp," %g",model->rho[i]);
		fprintf(fp, "\n");
	}
	
	if(model->label)
	{
		fprintf(fp, "label");
		for(int i=0;i<nr_class;i++)
			fprintf(fp," %d",model->label[i]);
		fprintf(fp, "\n");
	}

	if(model->probA) // regression has probA only
	{
		fprintf(fp, "probA");
		for(int i=0;i<nr_class*(nr_class-1)/2;i++)
			fprintf(fp," %g",model->probA[i]);
		fprintf(fp, "\n");
	}
	if(model->probB)
	{
		fprintf(fp, "probB");
		for(int i=0;i<nr_class*(nr_class-1)/2;i++)
			fprintf(fp," %g",model->probB[i]);
		fprintf(fp, "\n");
	}

	if(model->nSV)
	{
		fprintf(fp, "nr_sv");
		for(int i=0;i<nr_class;i++)
			fprintf(fp," %d",model->nSV[i]);
		fprintf(fp, "\n");
	}

	fprintf(fp, "SV\n");
	const double * const *sv_coef = model->sv_coef;
	const svm_node * const *SV = model->SV;

	for(int i=0;i<l;i++)
	{
		for(int j=0;j<nr_class-1;j++)
			fprintf(fp, "%.16g ",sv_coef[j][i]);

		const svm_node *p = SV[i];

		if(param.kernel_type == PRECOMPUTED)
			fprintf(fp,"0:%d ",(int)(p->value));
		else
			while(p->index != -1)
			{
				fprintf(fp,"%d:%.8g ",p->index,p->value);
				p++;
			}
		fprintf(fp, "\n");
	}

	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
	else return 0;
}

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
	int len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

//
// FSCANF helps to handle fscanf failures.
// Its do-while block avoids the ambiguity when
// if (...)
//    FSCANF();
// is used
//
#define FSCANF(_stream, _format, _var) do{ if (fscanf(_stream, _format, _var) != 1) return false; }while(0)
bool read_model_header(FILE *fp, svm_model* model)
{
	svm_parameter& param = model->param;
	char cmd[81];
	while(1)
	{
		FSCANF(fp,"%80s",cmd);

		if(strcmp(cmd,"svm_type")==0)
		{
			FSCANF(fp,"%80s",cmd);
			int i;
			for(i=0;svm_type_table[i];i++)
			{
				if(strcmp(svm_type_table[i],cmd)==0)
				{
					param.svm_type=i;
					break;
				}
			}
			if(svm_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown svm type.\n");
				return false;
			}
		}
		else if(strcmp(cmd,"kernel_type")==0)
		{		
			FSCANF(fp,"%80s",cmd);
			int i;
			for(i=0;kernel_type_table[i];i++)
			{
				if(strcmp(kernel_type_table[i],cmd)==0)
				{
					param.kernel_type=i;
					break;
				}
			}
			if(kernel_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown kernel function.\n");	
				return false;
			}
		}
		else if(strcmp(cmd,"degree")==0)
			FSCANF(fp,"%d",&param.degree);
		else if(strcmp(cmd,"gamma")==0)
			FSCANF(fp,"%lf",&param.gamma);
		else if(strcmp(cmd,"coef0")==0)
			FSCANF(fp,"%lf",&param.coef0);
		else if(strcmp(cmd,"nr_class")==0)
			FSCANF(fp,"%d",&model->nr_class);
		else if(strcmp(cmd,"total_sv")==0)
			FSCANF(fp,"%d",&model->l);
		else if(strcmp(cmd,"rho")==0)
		{
			int n = model->nr_class * (model->nr_class-1)/2;
			model->rho = Malloc(double,n);
			for(int i=0;i<n;i++)
				FSCANF(fp,"%lf",&model->rho[i]);
		}
		else if(strcmp(cmd,"label")==0)
		{
			int n = model->nr_class;
			model->label = Malloc(int,n);
			for(int i=0;i<n;i++)
				FSCANF(fp,"%d",&model->label[i]);
		}
		else if(strcmp(cmd,"probA")==0)
		{
			int n = model->nr_class * (model->nr_class-1)/2;
			model->probA = Malloc(double,n);
			for(int i=0;i<n;i++)
				FSCANF(fp,"%lf",&model->probA[i]);
		}
		else if(strcmp(cmd,"probB")==0)
		{
			int n = model->nr_class * (model->nr_class-1)/2;
			model->probB = Malloc(double,n);
			for(int i=0;i<n;i++)
				FSCANF(fp,"%lf",&model->probB[i]);
		}
		else if(strcmp(cmd,"nr_sv")==0)
		{
			int n = model->nr_class;
			model->nSV = Malloc(int,n);
			for(int i=0;i<n;i++)
				FSCANF(fp,"%d",&model->nSV[i]);
		}
		else if(strcmp(cmd,"SV")==0)
		{
			while(1)
			{
				int c = getc(fp);
				if(c==EOF || c=='\n') break;
			}
			break;
		}
		else
		{
			fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
			return false;
		}
	}

	return true;

}

svm_model *svm_load_model(const char *model_file_name)
{
	FILE *fp = fopen(model_file_name,"rb");
	if(fp==NULL) return NULL;

	char *old_locale = strdup(setlocale(LC_ALL, NULL));
	setlocale(LC_ALL, "C");

	// read parameters

	svm_model *model = Malloc(svm_model,1);
	model->rho = NULL;
	model->probA = NULL;
	model->probB = NULL;
	model->sv_indices = NULL;
	model->label = NULL;
	model->nSV = NULL;
	
	// read header
	if (!read_model_header(fp, model))
	{
		fprintf(stderr, "ERROR: fscanf failed to read model\n");
		setlocale(LC_ALL, old_locale);
		free(old_locale);
		free(model->rho);
		free(model->label);
		free(model->nSV);
		free(model);
		return NULL;
	}
	
	// read sv_coef and SV

	int elements = 0;
	long pos = ftell(fp);

	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	char *p,*endptr,*idx,*val;

	while(readline(fp)!=NULL)
	{
		p = strtok(line,":");
		while(1)
		{
			p = strtok(NULL,":");
			if(p == NULL)
				break;
			++elements;
		}
	}
	elements += model->l;

	fseek(fp,pos,SEEK_SET);

	int m = model->nr_class - 1;
	int l = model->l;
	model->sv_coef = Malloc(double *,m);
	int i;
	for(i=0;i<m;i++)
		model->sv_coef[i] = Malloc(double,l);
	model->SV = Malloc(svm_node*,l);
	svm_node *x_space = NULL;
	if(l>0) x_space = Malloc(svm_node,elements);

	int j=0;
	for(i=0;i<l;i++)
	{
		readline(fp);
		model->SV[i] = &x_space[j];

		p = strtok(line, " \t");
		model->sv_coef[0][i] = strtod(p,&endptr);
		for(int k=1;k<m;k++)
		{
			p = strtok(NULL, " \t");
			model->sv_coef[k][i] = strtod(p,&endptr);
		}

		while(1)
		{
			idx = strtok(NULL, ":");
			val = strtok(NULL, " \t");

			if(val == NULL)
				break;
			x_space[j].index = (int) strtol(idx,&endptr,10);
			x_space[j].value = strtod(val,&endptr);

			++j;
		}
		x_space[j++].index = -1;
	}
	free(line);

	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0)
		return NULL;

	model->free_sv = 1;	// XXX
	return model;
}

void svm_free_model_content(svm_model* model_ptr)
{
	if(model_ptr->free_sv && model_ptr->l > 0 && model_ptr->SV != NULL)
		free((void *)(model_ptr->SV[0]));
	if(model_ptr->sv_coef)
	{
		for(int i=0;i<model_ptr->nr_class-1;i++)
			free(model_ptr->sv_coef[i]);
	}

	free(model_ptr->SV);
	model_ptr->SV = NULL;

	free(model_ptr->sv_coef);
	model_ptr->sv_coef = NULL;

	free(model_ptr->rho);
	model_ptr->rho = NULL;

	free(model_ptr->label);
	model_ptr->label= NULL;

	free(model_ptr->probA);
	model_ptr->probA = NULL;

	free(model_ptr->probB);
	model_ptr->probB= NULL;

	free(model_ptr->sv_indices);
	model_ptr->sv_indices = NULL;

	free(model_ptr->nSV);
	model_ptr->nSV = NULL;
}

void svm_free_and_destroy_model(svm_model** model_ptr_ptr)
{
	if(model_ptr_ptr != NULL && *model_ptr_ptr != NULL)
	{
		svm_free_model_content(*model_ptr_ptr);
		free(*model_ptr_ptr);
		*model_ptr_ptr = NULL;
	}
}

void svm_destroy_param(svm_parameter* param)
{
	free(param->weight_label);
	free(param->weight);
}

const char *svm_check_parameter(const svm_problem *prob, const svm_parameter *param)
{
	// svm_type

	int svm_type = param->svm_type;
	if(svm_type != C_SVC &&
	   svm_type != NU_SVC &&
	   svm_type != ONE_CLASS &&
	   svm_type != EPSILON_SVR &&
	   svm_type != NU_SVR)
		return "unknown svm type";
	
	// kernel_type, degree
	
	int kernel_type = param->kernel_type;
	if(kernel_type != LINEAR &&
	   kernel_type != POLY &&
	   kernel_type != RBF &&
	   kernel_type != SIGMOID &&
	   kernel_type != PRECOMPUTED)
		return "unknown kernel type";

	if(param->gamma < 0)
		return "gamma < 0";

	if(param->degree < 0)
		return "degree of polynomial kernel < 0";

	// cache_size,eps,C,nu,p,shrinking

	if(param->cache_size <= 0)
		return "cache_size <= 0";

	if(param->eps <= 0)
		return "eps <= 0";

	if(svm_type == C_SVC ||
	   svm_type == EPSILON_SVR ||
	   svm_type == NU_SVR)
		if(param->C <= 0)
			return "C <= 0";

	if(svm_type == NU_SVC ||
	   svm_type == ONE_CLASS ||
	   svm_type == NU_SVR)
		if(param->nu <= 0 || param->nu > 1)
			return "nu <= 0 or nu > 1";

	if(svm_type == EPSILON_SVR)
		if(param->p < 0)
			return "p < 0";

	if(param->shrinking != 0 &&
	   param->shrinking != 1)
		return "shrinking != 0 and shrinking != 1";

	if(param->probability != 0 &&
	   param->probability != 1)
		return "probability != 0 and probability != 1";

	if(param->probability == 1 &&
	   svm_type == ONE_CLASS)
		return "one-class SVM probability output not supported yet";


	// check whether nu-svc is feasible
	
	if(svm_type == NU_SVC)
	{
		int l = prob->l;
		int max_nr_class = 16;
		int nr_class = 0;
		int *label = Malloc(int,max_nr_class);
		int *count = Malloc(int,max_nr_class);

		int i;
		for(i=0;i<l;i++)
		{
			int this_label = (int)prob->y[i];
			int j;
			for(j=0;j<nr_class;j++)
				if(this_label == label[j])
				{
					++count[j];
					break;
				}
			if(j == nr_class)
			{
				if(nr_class == max_nr_class)
				{
					max_nr_class *= 2;
					label = (int *)realloc(label,max_nr_class*sizeof(int));
					count = (int *)realloc(count,max_nr_class*sizeof(int));
				}
				label[nr_class] = this_label;
				count[nr_class] = 1;
				++nr_class;
			}
		}
	
		for(i=0;i<nr_class;i++)
		{
			int n1 = count[i];
			for(int j=i+1;j<nr_class;j++)
			{
				int n2 = count[j];
				if(param->nu*(n1+n2)/2 > min(n1,n2))
				{
					free(label);
					free(count);
					return "specified nu is infeasible";
				}
			}
		}
		free(label);
		free(count);
	}

	return NULL;
}

int svm_check_probability_model(const svm_model *model)
{
	return ((model->param.svm_type == C_SVC || model->param.svm_type == NU_SVC) &&
		model->probA!=NULL && model->probB!=NULL) ||
		((model->param.svm_type == EPSILON_SVR || model->param.svm_type == NU_SVR) &&
		 model->probA!=NULL);
}

void svm_set_print_string_function(void (*print_func)(const char *))
{
	if(print_func == NULL)
		svm_print_string = &print_string_stdout;
	else
		svm_print_string = print_func;
}
