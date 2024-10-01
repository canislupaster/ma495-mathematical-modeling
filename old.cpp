#include <algorithm>
#include <cwctype>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <map>
#include <numeric>
#include <string>
#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>

using namespace std;
using namespace Eigen;

struct ODE {
	vector<double> vars;
	ODE(vector<double>&& init): vars(init) {}
	virtual double compute_derivative(int var) = 0;

	void compute(double duration) {
		vector<double> dv(vars.size());
		vector<double> err(vars.size(),0);
		double dt = 1e-4;
		for (double t=0; t<duration; t+=dt) {
			for (int i=0; i<vars.size(); i++) {
				dv[i]=compute_derivative(i)*dt;
			}

			for (int i=0; i<vars.size(); i++) {
				double p=vars[i], a=dv[i]+err[i];
				vars[i]+=a;
				err[i] = a-(vars[i]-p);
			}
		}
	}
};

struct SIR: public ODE {
	double gamma, tau;
	SIR(double infected, double susceptible, double gamma, double tau):
		ODE({infected, susceptible}), gamma(gamma), tau(tau) {}
	double compute_derivative(int var) override {
		if (var==0) return -vars[0]*gamma + tau*vars[1]*vars[0];
		else return -tau*vars[1]*vars[0];
	}
};

struct NumericalCSVData {
	map<string, vector<optional<double>>> data;
	size_t nrow;

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
			auto it = data.emplace(hdr,vector<optional<double>>()).first;
			its.push_back(it);
		}

		nrow=tb.size()-1;
		for (int i=1; i<tb.size(); i++) {
			if (tb[i].size()!=data.size())
				throw runtime_error(format("row {} only has {} cols, expected {}", i,tb[i].size(),data.size()));

			for (int j=0; j<tb[i].size(); j++) {
				if (tb[i][j]=="X") {
					its[j]->second.emplace_back();
					continue;
				}

				char const* start=tb[i][j].c_str();
				char* end;
				double db = strtod(start, &end);
				if (end==start) throw runtime_error(format("no number (row {}, col {})", i,j));
				its[j]->second.push_back(db);
			}
		}
	}
};

struct Optim {
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

	vector<double> infected_counts;
	for (auto x: csv.data[*k]) infected_counts.push_back(*x/100);

	// auto& counts = csv.data["TOTALPATIENTS"];
	// double max_count = *max_element(counts.begin(), counts.end());
	// for (int i=0; i<infected_counts.size(); i++) infected_counts[i]/=counts[i];

	//di ~ x*a + y*b + xy*c
	VectorXd di(csv.nrow-1);
	VectorXd a(csv.nrow-1), b(csv.nrow-1), c(csv.nrow-1);

	double s=0;
	for (int i=0; i<csv.nrow-1; i++) {
		di(i)=infected_counts[i+1]-infected_counts[i];
		a(i)=-infected_counts[i];
		b(i)=(1-infected_counts[i]) * infected_counts[i];
		c(i)=-s*infected_counts[i];
		s+=infected_counts[i];
	}

	cout<<"a:\n"<<100*a<<"\nb:\n"<<100*b<<"\nc:\n"<<100*c<<"\ndi:\n"<<100*di<<"\n";
	
	VectorXd res(2); res<<0.3,0.3;
	Optim func {a,b,c,di};
	LevenbergMarquardt<Optim> lm(func);
	auto stat = lm.minimize(res);

	cout.precision(5);
	VectorXd resid(func.values());
	func(res, resid);
	cout<<"\nresiduals:\n"<<100*resid<<"\n";
	cout<<"\ngamma = "<<res(0)<<", tau = "<<res(1)<<"\n\n";

	SIR sir(infected_counts[0], 1-infected_counts[0], res(0), res(1));

	ofstream plot("./plot.gp");
	constexpr double step = 0.1;
	plot<<R"EOF(
		set title "SIR model approximation vs real data"
		set xlabel "Weeks"
		set ylabel "% of population"

		set xtics 1
		set ytics format "%.0f%%"

		set grid
	)EOF";

	plot<<"$SIR<<e\n";

	for (double t=1; t<=infected_counts.size(); t+=step) {
		plot<<t<<" "<<sir.vars[0]*100<<" "<<sir.vars[1]*100<<" "<<(1-sir.vars[0]-sir.vars[1])*100<<"\n";
		sir.compute(step);
	}

	plot<<"e\n$Data<<e\n";

	for (int i=0; i<infected_counts.size(); i++) {
		plot<<i+1<<" "<<infected_counts[i]*100<<"\n";
	}

	plot<<"e\n";
	
	plot<<"plot $SIR using 1:2 with lines title 'SIR model (infected)', ";
	plot<<"$SIR using 1:3 with lines title 'SIR model (susceptible)', ";
	plot<<"$SIR using 1:4 with lines title 'SIR model (recovered)', ";
	plot<<"$Data using 1:2 with lines title 'Data (infected)'\n";
}
