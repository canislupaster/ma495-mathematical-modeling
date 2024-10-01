#include <algorithm>
#include <cwctype>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>

using namespace std;
using namespace Eigen;

template<class T>
struct ODE {
	vector<T> vars,dv,err;
	ODE(vector<T>&& init, T zero): vars(init), dv(vars.size(),zero), err(vars.size(),zero) {}
	virtual T compute_derivative(int var) = 0;

	void compute(double duration) {
		double dt = 1e-4;
		for (double t=0; t<duration; t+=dt) {
			for (int i=0; i<vars.size(); i++) {
				dv[i]=compute_derivative(i)*dt;
			}

			for (int i=0; i<vars.size(); i++) {
				T p=vars[i], a=dv[i]+err[i];
				vars[i]+=a;
				err[i] = a-(vars[i]-p);
			}
		}
	}
};

// value, d_gamma, d_tau
struct SIR: public ODE<Vector3d> {
	double gamma, tau;
	SIR(double infected, double susceptible, double gamma, double tau):
		ODE({Vector3d(infected,0,0), Vector3d(susceptible,0,0)},
			Vector3d(0,0,0)), gamma(gamma), tau(tau) {}

	Vector3d compute_derivative(int var) override {
		auto trans = Vector3d(
			tau*vars[1](0)*vars[0](0),
			tau*vars[1](1)*vars[0](0) + tau*vars[1](0)*vars[0](1),
			tau*vars[1](2)*vars[0](0) + tau*vars[1](0)*vars[0](2) + vars[1](0)*vars[0](0)
		);

		if (var==0)
			return Vector3d(
				-vars[0](0)*gamma,
				-vars[0](0)-vars[0](1)*gamma,
				-vars[0](2)*gamma
			) + trans;
		else return -trans;
	}
};

struct NumericalCSVData {
	map<string, vector<string>> data;
	size_t nrow;

	vector<double> operator()(string const& k) const {
		vector<string> const& x = data.at(k);
		vector<double> out(x.size());

		for (int j=0; j<x.size(); j++) {
			char const* start=x[j].c_str();
			char* end;
			double db = strtod(start, &end);
			if (end==start) throw runtime_error(format("no number (col {}, row {})", k,j));
			out[j]=db;
		}

		return out;
	}

	NumericalCSVData(istream& in) {
		vector<vector<string>> tb;
		string s;
		while (getline(in, s)) {
			tb.emplace_back();
			char const* x = s.c_str();
			char const* p=x;
			do {
				if (*x=='\t' || *x==',' || !*x) {
					tb.back().emplace_back(p,x);
					p=x+1;
				}
			} while (*(x++));
		}

		vector<decltype(data.begin())> its;
		for (string& hdr: tb[0]) {
			auto it = data.emplace(hdr,vector<string>()).first;
			its.push_back(it);
		}

		nrow=tb.size()-1;
		for (int i=1; i<tb.size(); i++) {
			if (tb[i].size()!=data.size())
				throw runtime_error(format("row {} only has {} cols, expected {}", i,tb[i].size(),data.size()));

			for (int j=0; j<tb[i].size(); j++) {
				if (tb[i][j]=="X") its[j]->second.emplace_back();
				else its[j]->second.emplace_back(std::move(tb[i][j]));
			}
		}
	}
};

struct Optim {
	VectorXd infected;

	int inputs() {return 2;}
	int values() {return infected.rows()-1;}

	int operator()(VectorXd const& in, VectorXd& out) {
		SIR s(infected(0), 1-infected(0), in(0), in(1));
		for (int i=0; i<infected.size()-1; i++) {
			s.compute(1);
			out(i) = s.vars[0](0) - infected[i+1];
		}

		return 0;
	}

	int df(VectorXd const& in, MatrixXd& out) {
		SIR s(infected(0), 1-infected(0), in(0), in(1));
		for (int i=0; i<infected.size()-1; i++) {
			s.compute(1);
			out(i,0) = s.vars[0](1);
			out(i,1) = s.vars[0](2);
		}
		
		return 0;
	}
};

struct OptimFast {
	VectorXd a,b,c,di;

	int inputs() {return 2;}
	int values() {return a.rows();}

	int operator()(VectorXd const& in, VectorXd& out) {
		out = (a*in(0) + b*in(1) + in(0)*in(1)*c) - di;
		return 0;
	}

	int df(VectorXd const& in, MatrixXd& out) {
		out.col(0) = a + in(1)*c;
		out.col(1) = b + in(0)*c;
		return 0;
	}
};

int main(int argc, char** argv) {
	if (argc!=2) {
		throw new runtime_error("expected file path");
	}

	ifstream data(argv[1]);
	NumericalCSVData csv(data);

	optional<string> k=nullopt;
	vector<string> candidates = {
		"PERCENTPOSITIVE", "PERCENT POSITIVE",
		"%UNWEIGHTEDILI", "%UNWEIGHTED ILI"
	};

	for (string& c: candidates) {
		if (csv.data.contains(c)) {
			k=c;
			break;
		}
	}

	if (!k) throw runtime_error("data doesn't contain infected %");

	size_t n_train = 18, n_plot = 47;
	assert(n_plot>=csv.nrow);

	auto infected_counts = csv(*k);
	infected_counts.erase(infected_counts.begin()+n_plot, infected_counts.end());
	for (int i=0; i<n_plot; i++) infected_counts[i]/=100;

	VectorXd infected_vec(n_train);
	for (int i=0; i<n_train; i++) infected_vec(i)=infected_counts[i];

	VectorXd res(2); res<<0.3,0.3;
	auto fast_res = res;

	Optim func {infected_vec};
	LevenbergMarquardt<Optim> lm(func);

	lm.minimize(res);

	//di ~ x*a + y*b + xy*c
	VectorXd di(n_train-1);
	VectorXd a(n_train-1), b(n_train-1), c(n_train-1);

	double s=0;
	for (int i=0; i<n_train-1; i++) {
		di(i)=infected_counts[i+1]-infected_counts[i];
		a(i)=-infected_counts[i];
		b(i)=(1-infected_counts[i]) * infected_counts[i];
		c(i)=-s*infected_counts[i];
		s+=infected_counts[i];
	}

	OptimFast fast {a,b,c,di};
	LevenbergMarquardt<OptimFast> lm_fast(fast);
	lm_fast.minimize(fast_res);

	ofstream plot("./plot.gp");
	constexpr int nstep=10;

	double mse=0;
	auto compute_error = [&](VectorXd const& res, string const& name) {
		SIR sir(infected_counts[0], 1-infected_counts[0], res(0), res(1));

		plot<<"$"<<name<<"<<e\n";

		for (int i=0; i<n_plot; i++) {
			for (int j=0; j<nstep; j++) {
				plot<<i+double(j)/double(nstep)
					<<" "<<sir.vars[0](0)*100
					<<" "<<sir.vars[1](0)*100
					<<" "<<(1-sir.vars[0](0)-sir.vars[1](0))*100
					<<"\n";

				sir.compute(1.0/double(nstep));
			}
			
			double x=sir.vars[0](0)-infected_counts[i];
			mse+=x*x;
		}

		plot<<"e\n";

		mse/=n_plot;
		return mse;
	};

	ostringstream xtics;
	auto& year=csv.data["YEAR"], week=csv.data["WEEK"];
	for (int i=0; i<n_plot; i+=4) {
		xtics<<format("\"Wk {}\\n{}\" {}", week[i],year[i],i);
		if (i+5<n_plot) xtics<<", ";
	}

	plot<<format(R"EOF(
		set terminal pdfcairo lw 3.0
		set border lw 0.2
		set output "plot.pdf"
		set term pdfcairo font "Playfair Display bold,12"

		set title "SIR model approximation vs real data"
		set xlabel "Time"
		set ylabel "% of hospitalized"
		set yrange [0:100]
		set xrange [0:{1}]

		set xtics font ",8" ({2})
		set ytics format "%.0f%%"

		set arrow from {0}, graph 0 to {0}, graph 1 nohead
		set label "Training ends" at {0}+0.5, 20

		set grid
	)EOF", n_train, n_plot-1, xtics.view());

	plot<<"$Data<<e\n";

	for (int i=0; i<n_plot; i++) {
		plot<<i<<" "<<infected_counts[i]*100<<"\n";
	}

	plot<<"e\n";

	double err = compute_error(res, "SIR");
	double err_fast = compute_error(res, "SIRfast");
	
	plot<<"plot $SIR using 1:2 with lines title 'SIR model (infected)', ";
	plot<<"$SIR using 1:3 with lines title 'SIR model (susceptible)', ";
	plot<<"$SIR using 1:4 with lines title 'SIR model (recovered)', ";
	// plot<<"$SIR using 1:5 with lines title 'SIR model (old/fast)', ";
	plot<<"$Data using 1:2 with points title 'Data (infected)' ps 0.3 pt 7\n";

	cout.precision(5);
	cout<<"\n(ode) gamma = "<<res(0)<<", tau = "<<res(1)<<", mse="<<err<<"\n";
	cout<<"(fast) gamma = "<<fast_res(0)<<", tau = "<<fast_res(1)<<", mse="<<err_fast<<"\n\n";
}
