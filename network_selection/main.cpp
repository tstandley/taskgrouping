#include <iostream> 
#include <vector> 
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <cstdlib>
#include<numeric>

using namespace std; 
const float inf=9;

vector <float> read_line(stringstream &ssline) {

    vector<float> nline;
    while(!ssline.eof()) {
        char next = ssline.peek();
        if(next == '\t')
        {
            nline.push_back(inf);
            ssline.get();
        }
        else
        {
            float num;
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
    Model(vector<float> perf, float cost):performance(perf), cost(cost){}
    vector<float> performance;
    float cost;
    bool match(const Model &other) const {
        for (int i = 0; i < performance.size();i++)
            if((performance[i] == inf && other.performance[i]!=inf)||
               (performance[i] != inf && other.performance[i]==inf))
               return false;
        return true;
    }
    void remove_dim(int which_dim) {
        performance.erase(performance.begin()+which_dim);
    }
    // bool is_trivial() const{
    //     for (float i : performance)
    //         if (i < inf)
    //             return false;
    //     return true;
    // }

};

ostream & operator << (ostream & out, Model mod) {
    float sum=0;
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
                vector<float> infs({inf,inf,inf,inf,inf});
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

vector<float> score_solution (const vector<Model> &to_score,int size) {
    
    vector<float> score(size,inf);

    for (auto mod : to_score)
    {
        for(int i = 0; i < mod.performance.size();i++)
            score[i]=min(score[i],mod.performance[i]);
    }
    return score;
}

bool dominates(const vector<float>& first, const vector<float> & second) {
    for (int i = 0; i < first.size();i++)
        if (first[i] > second[i])
            return false;
    return true;
}

vector<Model> filter(const vector<Model> &candidates,const vector<float> & best_score, float remaining_budget) {
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
    auto a_score = score_solution(a,a[0].performance.size());
    auto b_score = score_solution(b,b[0].performance.size());
    float a_value = 0;
    float b_value = 0;
    for(auto i : a_score)a_value+=i;
    for(auto i : b_score)b_value+=i;
    if(a_value < b_value)
        return a;
    return b;
}

float get_sorting_score(const vector<float>& running_score,const Model & a) {
    float amax = -inf;
    for(int i = 0; i < running_score.size();i++)
        amax = max(amax,running_score[i]-a.performance[i]);
    return amax;
}

// float get_sorting_score(const vector<float>& running_score,const Model & a) {
//     float amax = 0;
//     for(int i = 0; i < running_score.size();i++)
//         amax += min(running_score[i],a.performance[i]);
//     return amax;
// }

vector<Model> get_best_networks(const vector<Model> &candidates_in, vector<Model> running_solution, float remaining_budget) {

    if (candidates_in.size()==0)
        return running_solution;

    vector<float> running_score = score_solution(running_solution,candidates_in[0].performance.size());


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
    vector<float> valid_values;
    for(int i =0; i < input.size();i++)
    {
        input[i].performance.push_back(inf);
        float val = input[i].performance[random_dimension];
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

int main() 
{
    //auto performances = get_model_performances("results.txt");
    auto performances = get_model_performances("results0.2.txt");
    auto performances_test_set = get_model_performances("results_test_set.txt");

    //performances = synthetic_performances(performances);
    //performances = synthetic_performances(performances);
    //performances = synthetic_performances(performances);
    performances = remove_task(performances,4);
    performances_test_set = remove_task(performances_test_set,4);

    for (auto line : performances_test_set) {
        cout << line << endl;
    }
    cout << "number_of_models= "<< performances.size()<< endl;

    for(float budget = 1; budget <= performances[0].performance.size();budget+=.5){
        auto solutiona = get_best_networks(performances,vector<Model>(),budget);
        
        auto solution = translate_scores_to_test_set(solutiona,performances_test_set);

        for(auto mod:solution) {
            cout << mod << endl;
        }
        cout << "budget=" << budget << " " << Model(score_solution(solution,solution[0].performance.size()),0) << endl;
    }
    return 0;
}