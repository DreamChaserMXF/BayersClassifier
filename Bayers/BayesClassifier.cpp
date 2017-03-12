#include "BayesClassifier.h"
#include "ConfusionMatrix.h"
#include <fstream>
#include <sstream>
#include <iostream>

using std::ifstream;
using std::istringstream;
using std::cin;
using std::cout;

BayesClassifier::BayesClassifier() :
	feature_num_(0), feature_name_(), feature_available_(),
	train_set_(), positive_set_(), negative_set_(),
	positive_ratio_(0.0), negative_ratio_(0.0),
	search_method_(BruteForceSearch), positive_kdtree_(), negative_kdtree_(),
	valid_feature_num_(0), coeff_(0.0)
{
	;
}

void BayesClassifier::SetFeatureNum(size_t feature_num)
{
	feature_num_ = feature_num;
}

void BayesClassifier::LoadSamples(const char *filename, bool has_title)
{
	ifstream in(filename);
	LoadSamples(in, has_title);
}

void BayesClassifier::LoadSamples(istream &in, bool has_title)
{
	train_set_.clear();
	string current_line;
	if(has_title)
	{
		std::getline(in, current_line);
	}
	size_t count = 0;
	size_t feature_num = feature_name_.size();
	Feature feature(feature_num, true);
	istringstream s_in;
	while(!in.eof())
	{
		getline(in, current_line);
		if(current_line.size() != 0)
		{
			s_in.str(current_line);
			s_in >> feature;
			train_set_.push_back(feature);
			++count;
		}
		s_in.clear();
	}
	UpdateAddressSet();
}

void BayesClassifier::LoadSamples(const vector<Feature> &samples)
{
	train_set_ = samples;
	UpdateAddressSet();
}

void BayesClassifier::RegularizeSamples(RegularizationMethod regu_method)
{
	if(Standardization == regu_method)
	{
		for(size_t i = 0; i < feature_num_; ++i)
		{
			// 对特征的每一维
			StandardizeSampleColumn(i);
		}
	}
	else if(Equalization == regu_method)
	{
		for(size_t i = 0; i < feature_num_; ++i)
		{
			// 对特征的每一维
			EqualizeSampleColumn(i);
		}
		// 由于最后一次对train_set_的改变是根据特征的最后一列进行排序，故这里将train_set_的顺序再按ID排好
		std::sort(train_set_.begin(), train_set_.end());

		// 由于EqualizeSampleColumn对train_set_中元素的顺序进行了改变，故要更新positive_set_与negative_set_
		UpdateAddressSet();
	}
}


void BayesClassifier::StandardizeSampleColumn(size_t col)
{
	// 求均值
	double mean = 0.0;
	for(size_t i = 0; i < train_set_.size(); ++i)
	{
		mean += train_set_[i].values_[col];
	}
	mean /= train_set_.size();
	// 求方差
	double variance = 0.0;
	for(size_t i = 0; i < train_set_.size(); ++i)
	{
		variance += (train_set_[i].values_[col] - mean) * (train_set_[i].values_[col] - mean);
	}
	// 规格化
	double standard_deviation = sqrt(variance);
	for(size_t i = 0; i < train_set_.size(); ++i)
	{
		train_set_[i].values_[col] = (train_set_[i].values_[col] - mean) / standard_deviation;
	}
}

void BayesClassifier::EqualizeSampleColumn(size_t col)
{
	// 1. 按col列进行排序
	std::sort(train_set_.begin(), train_set_.end(), FeatureColumnComparer(col));
	// 2. 用分布函数对该列的值作变换
	size_t total_count = train_set_.size();
	for(size_t i = 0; i < total_count; ++i)
	{
		train_set_[i].values_[col] = static_cast<double>(i) / total_count;
	}
}

void BayesClassifier::SetTrainMethod(SearchMethod search_method)
{
	search_method_ = search_method;
}

void BayesClassifier::Train()
{
	
	if(KDTreeSearch == search_method_)
	{
		// K-D Tree Training
		positive_kdtree_.SetDimension(feature_num_);
		positive_kdtree_.LoadFeatureName(feature_name_.begin());
		positive_kdtree_.LoadFeatureAvailable(feature_available_.begin());
		positive_kdtree_.LoadFeatureValidity(feature_available_.begin());
		positive_kdtree_.LoadSamples(positive_set_);
		positive_kdtree_.Train();

		negative_kdtree_.SetDimension(feature_num_);
		negative_kdtree_.LoadFeatureName(feature_name_.begin());
		negative_kdtree_.LoadFeatureAvailable(feature_available_.begin());
		negative_kdtree_.LoadFeatureValidity(feature_available_.begin());
		negative_kdtree_.LoadSamples(negative_set_);
		negative_kdtree_.Train();
	}
}

Feature::TYPE BayesClassifier::Classify(const Feature &feature) const
{
	double positive_probability = 0.0;
	double negative_probability = 0.0;
	size_t K = static_cast<size_t>(sqrt(static_cast<double>(train_set_.size())));	// K近邻的K
	double max_distance_square = 0.0;	// 找到的距feature最近的K个样本的最大距离
	// 检测feature为正例的概率（后验概率）
	// 1. 先计算先验概率
	if(KDTreeSearch == search_method_)
	{
		positive_kdtree_.FindNNeighbor(feature, K, vector<Feature*>(), max_distance_square);
	}
	else
	{
		FindNNeighbor(positive_set_, feature, K, vector<Feature*>(), max_distance_square);
	}
	double positive_volumn = coeff_ * pow(sqrt(max_distance_square), static_cast<int>(valid_feature_num_));	// 超球体的体积
	double positive_prior_probability = K / positive_volumn;
	double positive_joint_probability = positive_prior_probability * positive_ratio_;
	// 检测feature为负例的概率（后验概率）（其实应该是1-正例概率）
	if(KDTreeSearch == search_method_)
	{
		negative_kdtree_.FindNNeighbor(feature, K, vector<Feature*>(), max_distance_square);
	}
	else
	{
		FindNNeighbor(negative_set_, feature, K, vector<Feature*>(), max_distance_square);
	}
	double negative_volumn = coeff_ * pow(sqrt(max_distance_square), static_cast<int>(valid_feature_num_));	// 超球体的体积
	double negative_prior_probability = K / negative_volumn;
	double negative_joint_probability = negative_prior_probability * negative_ratio_; 
	
	// 判断
	if(positive_joint_probability > negative_joint_probability)
	{
		return Feature::TRUE;
	}
	else
	{
		return Feature::FALSE;
	}
}

//void BayesClassifier::Classify(const char *sample_file, const char *result_file, bool print_feature)
//{
//}


void BayesClassifier::SingleTest() const
{
	cout << "Single sample test\n";
	cout << "Input feature vector to classify, or \"quit\" to finish\n";

	string current_line;
	Feature feature(feature_num_, false);
	while(cin >> feature)
	{
		// 判断该特征属于哪个类别
		Feature::TYPE tag = Classify(feature);
		cout << tag << std::endl;
	}
	// 清除坏的标志位，并清空缓冲区
	cin.clear();
	cin.sync();
	//cin >> current_line;
	//cout << current_line << std::endl;
}

void BayesClassifier::CrossValidation(size_t fold) const
{
	size_t total_count = train_set_.size();
	size_t train_count = total_count * (fold - 1) / fold;
	int test_count = static_cast<int>(total_count / fold);
	ConfusionMatrix cmat_sum;
	cout << "\n" << fold << " cross validation:\n";
	for(size_t i = 0; i < fold; ++i)
	{
		cout << (i + 1) << ":\n";
		// 构造训练集
		vector<Feature> train_set(train_set_.begin(), train_set_.begin() + i * test_count);
		train_set.insert(train_set.end(), train_set_.begin() + (i + 1) * test_count, train_set_.end());
		// 通过训练集构造分类器
		BayesClassifier bayes;
		bayes.SetFeatureNum(feature_num_);
		bayes.LoadFeatureName(feature_name_.cbegin());
		bayes.LoadFeatureValidity(feature_available_.cbegin());
		bayes.LoadSamples(train_set);
		bayes.SetTrainMethod(search_method_);
		bayes.Train();
		// 对检测集(训练集的补集)进行检测
		ConfusionMatrix cmat;
		size_t count = 0;
		#pragma omp parallel for
		for(int j = 0; j < test_count; ++j)
		{
			#pragma omp atomic
			++count;
			cout << "\r\t\t\t\t\r" << 100 * count / test_count << "%... ";
			Feature::TYPE predicted_type = bayes.Classify(train_set_[i * test_count + j]);
			#pragma omp critical
			cmat.AddSample(train_set_[i * test_count + j].real_type_, predicted_type);
		}
		// 输出混淆矩阵
		cout << cmat << std::endl;
		cmat_sum += cmat;
	}
	// 输出总的混淆矩阵
	cout << "average result:\n";
	cout << cmat_sum << std::endl;
}

void BayesClassifier::UpdateAddressSet()
{
	positive_set_.clear();
	negative_set_.clear();
	size_t total_count = train_set_.size();
	for(size_t i = 0; i < total_count; ++i)
	{
		if(Feature::TRUE == train_set_[i].real_type_)
		{
			positive_set_.push_back(&train_set_[i]);
		}
		else
		{
			negative_set_.push_back(&train_set_[i]);
		}
	}
	positive_ratio_ = positive_set_.size() / static_cast<double>(total_count);
	negative_ratio_ = negative_set_.size() / static_cast<double>(total_count);
}

void BayesClassifier::FindNNeighbor(const vector<Feature*> &samples, const Feature &center, size_t n, vector<Feature*> &neighbors, double &max_distance_square) const
{
	FeatureDistanceComparer comparer(feature_available_, center);
	neighbors = samples;
	std::sort(neighbors.begin(), neighbors.end(), comparer);
	if(neighbors.size() > n)
	{
		neighbors.erase(neighbors.begin() + n, neighbors.end());
	}
	max_distance_square = comparer.distance(**(neighbors.rbegin()));
}