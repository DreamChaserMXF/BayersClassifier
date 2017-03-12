#ifndef BAYESCLASSIFIER
#define BAYESCLASSIFIER
#include "feature.h"
#include "KDTree.h"
#include <istream>
#include <algorithm>
using std::istream;
class BayesClassifier
{
public:
	//template<class _Iter1, class _Iter2>
	//BayesClassifier(size_t feature_num, _Iter1 feature_name, _Iter2 feature_validity);
	//template<class _Iter> void LoadFeatureName(_Iter first);
	//template<class _Iter> void LoadFeatureValidity(_Iter first);
	//LoadFeatureName(const vector<string> &feature_name);
	//LoadFeatureValidity(const vector<bool> &feature_validity);

	enum SearchMethod{KDTreeSearch, BruteForceSearch};
	enum RegularizationMethod{Standardization, Equalization};

	BayesClassifier();
	void SetFeatureNum(size_t feature_num);
	template<class _Iter> void LoadFeatureName(_Iter first);
	template<class _Iter> void LoadFeatureValidity(_Iter first);
	void LoadSamples(const char *filename, bool has_title);
	void LoadSamples(istream &in, bool has_title);
	void LoadSamples(const vector<Feature> &samples);

	void RegularizeSamples(RegularizationMethod regu_method);

	void SetTrainMethod(SearchMethod search_method);
	void Train();
	Feature::TYPE Classify(const Feature &feature) const;
	//void Classify(const char *sample_file, const char *result_file, bool print_feature = false);

	void SingleTest() const;
	void CrossValidation(size_t fold) const;
private:
	BayesClassifier(const BayesClassifier &);

	void UpdateAddressSet();

	void StandardizeSampleColumn(size_t col);
	void EqualizeSampleColumn(size_t col);
	void FindNNeighbor(const vector<Feature*> &samples, const Feature &center, size_t n, vector<Feature*> &neighbors, double &max_distance_square) const;

	size_t feature_num_;
	vector<string> feature_name_;
	vector<bool> feature_available_;
	vector<Feature> train_set_;
	vector<Feature*> positive_set_;
	vector<Feature*> negative_set_;
	double positive_ratio_;
	double negative_ratio_;

	SearchMethod search_method_;
	KDTree positive_kdtree_;
	KDTree negative_kdtree_;

	// 预先计算量，为了计算高维球体的体积
	size_t valid_feature_num_;
	double coeff_;	
};

//template<class _Iter1, class _Iter2>
//BayesClassifier::BayesClassifier(size_t feature_num, _Iter1 feature_name, _Iter2 feature_validity) :
//	feature_num_(feature_num),
//	feature_name_(feature_name, feature_name + feature_num), 
//	feature_validity_(feature_validity, feature_validity + feature_num),
//	train_set_(), positive_set_(), negative_set_(),
//	positive_ratio_(0.0), negative_ratio_(0.0)
//{
//	;
//}

template<class _Iter> void BayesClassifier::LoadFeatureName(_Iter first)
{
	feature_name_.assign(first, first + feature_num_);
}
template<class _Iter> void BayesClassifier::LoadFeatureValidity(_Iter first)
{
	feature_available_.assign(first, first + feature_num_);
	valid_feature_num_ = std::count(feature_available_.begin(), feature_available_.end(), true);
	const double PI = 3.141592653;
	size_t k = valid_feature_num_ / 2;
	
	if(2 * k == valid_feature_num_)	// even number
	{
		coeff_ = 1.0;
		for(size_t i = 1; i <= k; ++i)
		{
			coeff_ *= PI / i;
		}
	}
	else	// odd number
	{
		coeff_ = 2.0;
		for(size_t i = 1; i <= k; ++i)
		{
			coeff_ *= 2 * PI / ((i << 1) + 1);
		}
	}
}
#endif