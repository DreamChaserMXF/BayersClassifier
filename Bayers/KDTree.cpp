#include "KDTree.h"
#include <algorithm>


KDTree::KDTree() :
	dimension_(0), feature_name_(), feature_validity_(),
	samples_(),
	left_branch_(NULL), right_branch_(NULL), branch_column_(0), branch_midvalue_(0.0)
{
	;
}
KDTree::~KDTree()
{
	if(left_branch_)
	{
		delete left_branch_;
		delete right_branch_;
	}
}
void KDTree::SetDimension(size_t dimension)
{
	dimension_ = dimension;
}

void KDTree::LoadSamples(const vector<Feature*> &samples)
{
	samples_ = samples;
}

void KDTree::Train()
{
	// 1. 若只有1个样本，则结束
	if(samples_.size() <= 1)
	{
		return;
	}
	// 2. 在有效列中找出方差最大的一列（若无有效列，则结束）
	double max_variance = 0.0;
	int max_var_column = -1;
	for(size_t i = 0; i < dimension_; ++i)
	{
		if(feature_validity_[i])
		{
			// 计算该列均值
			double sum = 0.0;
			for(vector<Feature*>::const_iterator iter = samples_.begin();
				iter != samples_.end(); ++iter)
			{
				sum += (*iter)->values_[i];
			}
			double average = sum / samples_.size();
			// 计算方差
			sum = 0.0;
			for(vector<Feature*>::const_iterator iter = samples_.begin();
				iter != samples_.end(); ++iter)
			{
				sum += ((*iter)->values_[i] - average) * ((*iter)->values_[i] - average);
			}
			// 这里sum没有除以样本数量，故不是严格的方差，但也可以用来比较
			if(sum > max_variance)	// 这里不用>=而用>，是为了处理，当只有一个样本时，便不需要再往下分的情况
			{
				max_variance = sum;
				max_var_column = i;
			}
		}
	}
	if(-1 == max_var_column)
	{
		return;
	}
	branch_column_ = max_var_column;
	// 3. 对方差最大的一列进行排序，选出中位数
	std::sort(samples_.begin(), samples_.end(), FeatureColumnComparer(branch_column_));
	size_t half_size = samples_.size() / 2;
	if((half_size << 1) == samples_.size())
	{
		branch_midvalue_ = (samples_[half_size - 1]->values_[branch_column_]
		+ samples_[half_size]->values_[branch_column_]) / 2.0;
	}
	else
	{
		branch_midvalue_ = samples_[half_size]->values_[branch_column_];
	}
	// 4. 以中位数为分叉点，对当前树进行分支
	vector<bool> new_feature_validity = feature_validity_;
	new_feature_validity[branch_column_] = false;

	left_branch_ = new KDTree();
	left_branch_->SetDimension(dimension_);
	left_branch_->LoadFeatureName(feature_name_.begin());
	left_branch_->LoadFeatureAvailable(feature_available_.begin());
	left_branch_->LoadFeatureValidity(new_feature_validity.begin());
	left_branch_->LoadSamples(vector<Feature*>(samples_.begin(), samples_.begin() + half_size));
	left_branch_->Train();

	right_branch_ = new KDTree();
	right_branch_->SetDimension(dimension_);
	right_branch_->LoadFeatureName(feature_name_.begin());
	right_branch_->LoadFeatureAvailable(feature_available_.begin());
	right_branch_->LoadFeatureValidity(new_feature_validity.begin());
	right_branch_->LoadSamples(vector<Feature*>(samples_.begin() + half_size, samples_.end()));
	right_branch_->Train();
}

void KDTree::FindNNeighbor(const Feature &center, size_t n, vector<Feature*> &neighbors, double &max_distance_square) const
{
	// 如果是叶节点，则根据距离排序，寻找
	if(NULL == left_branch_)
	{
		FeatureDistanceComparer comparer(feature_available_, center);
		neighbors.insert(neighbors.end(), samples_.begin(), samples_.end());
		std::sort(neighbors.begin(), neighbors.end(), comparer);
		if(neighbors.size() > n)	// 这里是有必要的，否则可能会报迭代器异常
		{
			neighbors.erase(neighbors.begin() + n, neighbors.end());
		}
		max_distance_square = comparer.distance(**(neighbors.rbegin()));
	}
	else if(center.values_[branch_column_] < branch_midvalue_)
	{
		left_branch_->FindNNeighbor(center, n, neighbors, max_distance_square);
		if( neighbors.size() < n
			|| (center.values_[branch_column_] - branch_midvalue_) * (center.values_[branch_column_] - branch_midvalue_) < max_distance_square)
		{
			right_branch_->FindNNeighbor(center, n, neighbors, max_distance_square);
		}
	}
	else
	{
		right_branch_->FindNNeighbor(center, n, neighbors, max_distance_square);
		if( neighbors.size() < n
			|| (center.values_[branch_column_] - branch_midvalue_) * (center.values_[branch_column_] - branch_midvalue_) < max_distance_square)
		{
			left_branch_->FindNNeighbor(center, n, neighbors, max_distance_square);
		}
	}

}