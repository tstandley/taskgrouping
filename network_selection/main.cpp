#include <iostream> 
#include <vector> 
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <string>
#include <ctime>
using namespace std; 
const double inf=99;

vector <double> read_line(stringstream &ssline) {

    vector<double> nline;
    while(!ssline.eof()) {
        char next = ssline.peek();
        if(next == '\t')
        {
            nline.push_back(inf);
            ssline.get();
        }
        else
        {
            double num;
            if(ssline >> num) {
                ssline.get();
                nline.push_back(num);
            }
            else
            {
                nline.push_back(inf);
            }
            
        }
    }
    return nline;
}
class Model
{
    public:
    Model(vector<double> perf, double cost):performance(perf), cost(cost){}
    Model():cost(-1){}
    vector<double> performance;
    double cost;
    bool match(const Model &other) const {
        for (int i = 0; i < performance.size();i++)
            if((performance[i] == inf && other.performance[i]!=inf)||
               (performance[i] != inf && other.performance[i]==inf))
               return false;
        return cost==other.cost;
    }
    void remove_dim(int which_dim) {
        performance.erase(performance.begin()+which_dim);
    }
    double total_score(){
        double sum=0;
        for(auto i : performance)
            sum+=i;
        return sum;
    }

    int rank() const {
        int r=0;
        for (double i : performance)
            if (i < inf)
                r++;
        return r;
    }
    

};

ostream & operator << (ostream & out, Model mod) {
    double sum=0;
    for(auto i : mod.performance) {
        out << i << ", ";
        sum+=i;
    }
    if(mod.cost==0)
    {
        out << "=" << sum;
    }
    else
    {
        out << mod.cost;        
    }
    return out;
}

vector<Model> get_model_performances(string filename) {
    ifstream file(filename);
    
    string line;

    vector<Model> ret;
    while(std::getline(file,line))
    {  
        stringstream ssline(line);
        char next;
        if (ret.size()==0) {
            auto nline = read_line(ssline);
            
            for(int i = 0; i < nline.size();i++)
            {
                vector<double> infs({inf,inf,inf,inf,inf});
                infs[i]=nline[i];
                ret.push_back(Model(infs,0.5));
            }
        } else {
            auto nline = read_line(ssline);
            ret.push_back(Model(nline,1));
        }
    }

    return ret;
}

vector<double> score_solution (const vector<Model> &to_score,int size=-1) {
    
    if(size==-1) size = to_score[0].performance.size();
    vector<double> score(size,inf);

    for (auto mod : to_score)
    {
        for(int i = 0; i < mod.performance.size();i++)
            score[i]=min(score[i],mod.performance[i]);
    }
    return score;
}

bool dominates(const vector<double>& first, const vector<double> & second) {
    for (int i = 0; i < first.size();i++)
        if (first[i] > second[i])
            return false;
    return true;
}

vector<Model> filter(const vector<Model> &candidates,const vector<double> & best_score, double remaining_budget) {
    vector<Model> to_return;
    for(auto candidate : candidates) {
        if (candidate.cost <= remaining_budget && !dominates(best_score,candidate.performance) )
            to_return.push_back(candidate);
    }
    return to_return;
}

vector<Model> better(const vector<Model> & a, const vector<Model> & b) {
    if (a.size()==0)
        return b;
    auto a_score = score_solution(a);
    auto b_score = score_solution(b);
    double a_value = 0;
    double b_value = 0;
    for(auto i : a_score)a_value+=i;
    for(auto i : b_score)b_value+=i;
    if(a_value < b_value)
        return a;
    return b;
}

double get_sorting_score(const vector<double>& running_score,const Model & a) {
    double amax = -inf;
    for(int i = 0; i < running_score.size();i++)
        amax = max(amax,running_score[i]-a.performance[i]);
    return amax;
}

// double get_sorting_score(const vector<double>& running_score,const Model & a) {
//     double amax = 0;
//     for(int i = 0; i < running_score.size();i++)
//         amax += min(running_score[i],a.performance[i]);
//     return amax;
// }

vector<Model> get_best_networks(const vector<Model> &candidates_in, vector<Model> running_solution, double remaining_budget) {

    if (candidates_in.size()==0)
        return running_solution;

    vector<double> running_score = score_solution(running_solution,candidates_in[0].performance.size());


    auto candidates = filter(candidates_in,running_score,remaining_budget);
    if (candidates.size()==0)
        return running_solution;
    std::sort(candidates.begin(),candidates.end(),
        [&running_score](const Model & a, const Model & b)-> bool{
            return get_sorting_score(running_score,a) < get_sorting_score(running_score,b);
        }
    );
    

    vector<Model> best_solution = running_solution;


    while(!candidates.empty()) {
        
        running_solution.push_back(candidates.back());
        candidates.pop_back();
        auto best_below = get_best_networks(candidates,running_solution,remaining_budget-running_solution.back().cost);
        
        running_solution.pop_back();
        best_solution = better(best_solution,best_below);
    }
    return best_solution;

}

vector<Model> synthetic_performances(vector<Model>input) {

    int num_tasks=input[0].performance.size();

    int random_dimension = rand()%num_tasks;
    vector<double> valid_values;
    for(int i =0; i < input.size();i++)
    {
        input[i].performance.push_back(inf);
        double val = input[i].performance[random_dimension];
        if(val < inf)
            valid_values.push_back(val);
    }
    int fixed_input_size=input.size();
    for(int i =5; i < fixed_input_size;i++)
    {
        input.push_back(input[i]);
        input.back().performance.back()=valid_values[rand()%valid_values.size()];
        for(int ii =0;ii < input.back().performance.size();ii++){
            input.back().performance[ii]*=1.05;
        }
    }
    return input;}


vector<Model> translate_scores_to_test_set(const vector<Model> &solution,const vector<Model> &test_performances){
    vector<Model> ret;
    for (Model mod : solution) {
        for(Model mod2 : test_performances) {
            if (mod.match(mod2)){
                ret.push_back(mod2);
                //break;
            }
        }
    }
    return ret;
}

vector<Model> remove_task(vector<Model> performances,int task) {
    vector<Model> ret;

    for(auto i : performances) {
        if (i.performance[task]==inf) {
            i.remove_dim(task);
            ret.push_back(i);
        }
    }

    return ret;
}

bool subset(Model &a, Model&b) {
    if (a.performance.size() != b.performance.size())
        return false;
    if (a.rank() > b.rank())
        return false;
    for (int i=0;i<a.performance.size();i++) {
        if (a.performance[i] < inf)
            if(b.performance[i] == inf)
                return false;
    }
    return true;
}

vector <Model> higher_order_approximation(vector<Model> performances) {
    

    vector<Model> pairs_models;
    for (auto a : performances)
        if(a.rank() ==2)
            pairs_models.push_back(a);

    vector<Model> new_models;
    for (auto a : performances)
        if(a.rank() <=2)
            new_models.push_back(a);
        else {
            Model new_model({0,0,0,0,0},1);
            vector<int> count({0,0,0,0,0});
            for (auto pair_model:pairs_models)
                if (subset(pair_model,a)) {
                    for(int i = 0; i < pair_model.performance.size();i++)
                        if (pair_model.performance[i]!=inf){
                            new_model.performance[i]+=pair_model.performance[i];
                            count[i]++;
                        }
                }
            for(int i=0;i<new_model.performance.size();i++){
                if (count[i]==0)
                    new_model.performance[i]=inf;
                else
                    new_model.performance[i]/=count[i];
            }
                
            //cout << new_model << endl;
            new_models.push_back(new_model);
        }

    return new_models;
}

vector <Model> just_pairs(vector<Model> performances) {
    vector<Model> new_models;
    for (auto a : performances)
        if(a.rank() <=2 || a.rank()==5)
            new_models.push_back(a);
        

    return new_models;
}

vector<double> get_mins(vector<Model> performances) {
    vector<double> mins({inf,inf,inf,inf,inf});
    for (auto a : performances){
        for(int i =0;i<a.performance.size();i++) {
            if(a.performance[i] != inf) {
                if(mins[i] > a.performance[i]) mins[i]= a.performance[i];
            }

        }
    }
    return mins;
}

vector<double> get_maxes(vector<Model> performances) {
    vector<double> maxes({0,0,0,0,0});
    for (auto a : performances){
        for(int i =0;i<a.performance.size();i++) {
            if(a.performance[i] != inf) {
                if(maxes[i] < a.performance[i])
                 maxes[i]=a.performance[i];
            }

        }
    }
    return maxes;
}

vector <Model> scale_values(vector<Model> performances) {

    auto mins = get_mins(performances);
    auto maxes = get_maxes(performances);
    
    for(int i = 0; i < maxes.size();i++) {
        cout << maxes[i] << ' ';
    }
    cout << endl;

    for(int i = 0; i < maxes.size();i++) {
        cout << mins[i] << ' ';
    }
    cout << endl;

    vector <Model> scaled_models;
    for (auto a : performances){
        for(int i =0;i<a.performance.size();i++) {
            if (a.performance[i]!=inf)
                a.performance[i]=(a.performance[i]-mins[i])/(maxes[i]-mins[i]);
        }
        scaled_models.push_back(a);
    }
    return scaled_models;
}

vector<double> compute_and_print(vector<Model> &performances, vector<Model> & performances_test_set, bool print=false) {
    if(print)
        cout << "number_of_models= "<< performances.size()<< endl;

    vector<double> perfs;
    for(double budget = 1; budget <= performances[0].performance.size();budget+=.5){
        auto solutiona = get_best_networks(performances,vector<Model>(),budget);
        
        auto solution = translate_scores_to_test_set(solutiona,performances_test_set);
        //auto solution=solutiona;
        
        Model sol(score_solution(solution,solution[0].performance.size()),0);
        if (print){
            for(auto mod:solution) {
                cout << mod << endl;
            }    
            cout << "budget=" << budget << " " << sol << endl;
        }
        perfs.push_back(sol.total_score());
    }
    if (print)
        for(auto p:perfs)
            cout << p << endl;
    return perfs;
}


double do_one_random(vector<Model> test_perfs, double budget){

    while(true) {
        double remaining_budget=budget;

        vector<Model> random_sol;
        vector<bool> used(test_perfs.size(), false);
        int used_count=0;
        while(true)
        {
            int index;
            do {
                index=rand()%test_perfs.size();
            } while(used[index]);
            used[index]=true;
            used_count++;

            if(used_count == test_perfs.size()) {
                used_count=0;
                for(int i =0;i<used.size();i++)
                    used[i]=false;
                random_sol.resize(0);
                remaining_budget=budget;
            }

            if(test_perfs[index].cost <=remaining_budget) {
                random_sol.push_back(test_perfs[index]);
                remaining_budget-=test_perfs[index].cost;
            }
            
            if (remaining_budget < 0.5)
                break;
            
        }
        Model sol(score_solution(random_sol),0); 
        double total_score=sol.total_score();
        if(total_score < inf) {
            // if(budget==1) {
            //     if(total_score < 0.50276 || total_score > 0.50279)
            //         cout << total_score << ' ' << random_sol.size() << endl;
            // }

            return total_score;
        }


    }

}



// vector<double> do_random(vector<Model> test_perfs) {

//     vector<double> totals;
//     int num_times=3000;
//     for(int i=0;i<num_times;i++) {
//         auto rand_models=get_random_models(test_perfs);
//         auto rand_perf=compute_and_print(rand_models,test_perfs);
//         if (totals.size()==0)
//             totals=rand_perf;
//         else
//             for(int ii=0;ii<rand_perf.size();ii++)
//                 totals[ii]+=rand_perf[ii];
//     }
//     for(int i = 0; i < totals.size();i++)
//         totals[i]/=num_times;
//     return totals;
// }


vector<double> do_random(vector<Model> test_perfs) {

    vector<double> totals({0,0,0,0,0,0,0,0,0});
    int num_times=1000000;
    for(int i=0;i<num_times;i++) {
        int c=0;
        
        for(double budget = 1;budget<=5;budget+=.5) {
            auto val = do_one_random(test_perfs,budget);
            //cout << c << endl;
            totals[c]+=val;
            c++;
        }
    }

    for(int i = 0; i < totals.size();i++)
        totals[i]/=(num_times);
    return totals;
}

vector<double> do_worst(vector<Model> test_perfs) {

    vector<double> totals({0,0,0,0,0,0,0,0,0});
    int num_times=10000000;
    for(int i=0;i<num_times;i++) {
        int c=0;
        for(double budget = 1;budget<=5;budget+=.5) {
            auto val = do_one_random(test_perfs,budget);
            if (totals[c] < val)
                totals[c] = val;
            c++;
        }
    }
    
    return totals;
}

void do_setting(string val_set_file, string test_set_file, string esa_set_file) 
{
    
    auto val_perfs = get_model_performances(val_set_file);
    //val_perfs=scale_values(val_perfs);
    auto test_perfs = get_model_performances(test_set_file);
    //test_perfs=scale_values(test_perfs);
    auto esa_perfs = get_model_performances(esa_set_file);
    //esa_perfs=scale_values(esa_perfs);
    
    auto hoa_perfs = higher_order_approximation(val_perfs);
    auto pair_perfs = just_pairs(val_perfs);


    cout << "optimal = ";
    auto perfs = compute_and_print(val_perfs,test_perfs);
    cout << '[';
    for(auto p:perfs)
        cout << p << ", ";
    cout << "\b\b]" << endl;
    
    cout << "esa = ";
    perfs = compute_and_print(esa_perfs,test_perfs);
    cout << '[';
    for(auto p:perfs)
        cout << p << ", ";
    cout << "\b\b]" << endl;
        
    cout << "hoa = ";
    perfs = compute_and_print(hoa_perfs,test_perfs);
    cout << '[';
    for(auto p:perfs)
        cout << p << ", ";
    cout << "\b\b]" << endl;

    // cout << "pairs = ";
    // perfs = compute_and_print(pair_perfs,test_perfs);
    // cout << '[';
    // for(auto p:perfs)
    //     cout << p << ", ";
    // cout << "\b\b]" << endl;
    
    cout << "random = ";
    perfs = do_random(test_perfs);
    cout << '[';
    for(auto p:perfs)
        cout << p << ", ";
    cout << "\b\b]" << endl;

    cout << "worst = ";
    perfs = do_worst(test_perfs);
    cout << '[';
    for(auto p:perfs)
        cout << p << ", ";
    cout << "\b\b]" << endl;
        
}

int main() {
    srand (time(NULL));
    cout << "\nsetting 1" << endl;
    do_setting("results.txt","results_test.txt","results_20.txt");
    cout << "\nsetting 2" << endl;
    do_setting("results_large.txt","results_large_test.txt","results_large_20.txt");
    cout << "\nsetting 3" << endl;
    do_setting("results_small_data.txt","results_small_data_test.txt","results_small_data_at4.txt");
    cout << "\nsetting 4" << endl;
    do_setting("results_alt.txt","results_alt_test.txt","results_alt_20.txt");
    return 0;
}