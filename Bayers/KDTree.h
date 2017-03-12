#ifndef KDTREE_H
#define KDTREE_H

#include "feature.h"
#include <vector>
using std::vector;

//class KDTreeNode
//{
//public:
//	KDTreeNode();
//
//};

class KDTree
{
public:
	KDTree();
	~KDTree();
	void SetDimension(size_t dimension);
	template<class _Iter> void LoadFeatureName(_Iter first);
	template<class _Iter> void LoadFeatureAvailable(_Iter first);
	template<class _Iter> void LoadFeatureValidity(_Iter first);
	void LoadSamples(const vector<Feature*> &samples);
	void Train();
	void FindNNeighbor(const Feature &center, size_t n, vector<Feature*> &neighbors, double &max_distance_square) const;
private:
	// 禁止复制
	KDTree(const KDTree &);	
	KDTree& operator = (const KDTree &);

	size_t dimension_;
	vector<string> feature_name_;
	vector<bool> feature_available_;	// 真正可用的feature
	vector<bool> feature_validity_;		// 真正可用的feature - 祖先节点用过的feature
	vector<Feature*> samples_;

	KDTree *left_branch_;
	KDTree *right_branch_;
	size_t branch_column_;
	double branch_midvalue_;
};

template<class _Iter> void KDTree::LoadFeatureName(_Iter first)
{
	feature_name_.assign(first, first + dimension_);
}

template<class _Iter> void KDTree::LoadFeatureAvailable(_Iter first)
{
	feature_available_.assign(first, first + dimension_);
}

template<class _Iter> void KDTree::LoadFeatureValidity(_Iter first)
{
	feature_validity_.assign(first, first + dimension_);
}

#endif