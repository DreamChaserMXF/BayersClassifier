#ifndef FEATURE_H
#define FEATURE_H
#include <string>
#include <vector>
#include <istream>
#include <ostream>
using std::string;
using std::vector;
using std::istream;
using std::ostream;

class Feature
{
public:
	
	enum TYPE{FALSE = 1, TRUE = -1, VOID = 2};

	Feature(int attr_count, bool isLabeled);
	string id_;
	vector<double> values_;
	TYPE real_type_;					// 实际类别
	TYPE predicted_type_;				// 预测类别(只能由分类器赋值)
	bool isLabeled_;					// 该Feature是否有过标注

	bool operator < (const Feature &right);
};

ostream& operator << (ostream &out, const Feature &feature);
istream& operator >> (istream &in, Feature &feature);


class FeatureColumnComparer
{
public:
	FeatureColumnComparer(size_t column) : column_(column)
	{
		;
	}
	bool operator ()(const Feature &left, const Feature &right) const
	{
		return left.values_[column_] < right.values_[column_];
	}
	bool operator ()(const Feature *left, const Feature *right) const
	{
		return left->values_[column_] < right->values_[column_];
	}

private:
	size_t column_;
};

class FeatureDistanceComparer
{
public:
	FeatureDistanceComparer(const vector<bool> &feature_available, const Feature &center) : 
	  feature_available_(feature_available), center_(center)
	{
		;
	}

	//static double distance(const vector<bool> &feature_available, const Feature &feature1, const Feature &feature2)
	//{
	//	double sum = 0.0;
	//	for(size_t i = 0; i < feature_available.size(); ++i)
	//	{
	//		if(feature_available[i])
	//		{
	//			sum += (feature1.values_[i] - feature2.values_[i]) * (feature1.values_[i] - feature2.values_[i]);
	//		}
	//	}
	//	return sum;
	//}

	double distance(const Feature &feature) const
	{
		double sum = 0.0;
		for(size_t i = 0; i < feature_available_.size(); ++i)
		{
			if(feature_available_[i])
			{
				sum += (feature.values_[i] - center_.values_[i]) * (feature.values_[i] - center_.values_[i]);
			}
		}
		return sum;
	}

	bool operator ()(const Feature *left, const Feature *right) const
	{
		return distance(*left) < distance(*right);
		//return distance(feature_available_, *left, center_) < distance(feature_available_, *right, center_);
	}

private:
	vector<bool> feature_available_;
	Feature center_;
};

#endif