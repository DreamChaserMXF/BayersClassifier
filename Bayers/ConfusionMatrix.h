#ifndef CONFUSIONMATRIX_H
#define CONFUSIONMATRIX_H
#include <iostream>
#include "feature.h"

using std::ostream;

class ConfusionMatrix
{
public:
	ConfusionMatrix() : 
		true_positive_(0), true_negative_(0),
		false_positive_(0), false_negative_(0)
	{}
	ConfusionMatrix(size_t true_positive, size_t true_negative, size_t false_positive, size_t false_negative) : 
					true_positive_(true_positive), true_negative_(true_negative),
					false_positive_(false_positive), false_negative_(false_negative)
	{}
	void AddSample(Feature::TYPE real_type, Feature::TYPE predicted_type)
	{
		if(Feature::TRUE == real_type && Feature::TRUE == predicted_type)
		{
			++true_positive_;
		}
		else if(Feature::FALSE == real_type && Feature::FALSE == predicted_type)
		{
			++true_negative_;
		}
		else if(Feature::TRUE == real_type && Feature::FALSE == predicted_type)
		{
			++false_negative_;
		}
		else if(Feature::FALSE == real_type && Feature::TRUE == predicted_type)
		{
			++false_positive_;
		}
	}

	ConfusionMatrix& operator += (const ConfusionMatrix &m)
	{
		true_positive_ += m.true_positive_;
		true_negative_ += m.true_negative_;
		false_positive_ += m.false_positive_;
		false_negative_ += m.false_negative_;
		return *this;
	}
	size_t true_positive_;
	size_t true_negative_;
	size_t false_positive_;
	size_t false_negative_;
};

ostream& operator << (ostream &out, const ConfusionMatrix &c_mat)
{
	out << "\t\t\tactual value" << '\n';
	out << "\t\t\tp (" << Feature::TRUE << ")\tn (" << Feature::FALSE << ")" << '\n';
	out << "prediction:\tp (" << Feature::TRUE << ")\t" << c_mat.true_positive_ << "\t" << c_mat.false_positive_ << '\n';
	out << "\t\tn (" << Feature::FALSE << ")\t" << c_mat.false_negative_ << "\t" << c_mat.true_negative_ << '\n';
	out << '\n';
			
	double precision = 0.0;
	double accuracy = 0.0;
	double recall = 0.0;
	double devider = 0.0;
	int total_sample = c_mat.true_positive_ + c_mat.true_negative_ + c_mat.false_positive_ + c_mat.false_negative_;

	if(c_mat.true_positive_ + c_mat.false_positive_ > 0)
	{
		precision = static_cast<double>(c_mat.true_positive_) / static_cast<double>(c_mat.true_positive_ + c_mat.false_positive_);
	}
	else
	{
		precision = 1.0;
	}
	
	if(total_sample > 0)
	{
		accuracy = static_cast<double>(c_mat.true_positive_ + c_mat.true_negative_) /
			static_cast<double>(c_mat.true_positive_ + c_mat.true_negative_ + c_mat.false_positive_ + c_mat.false_negative_);
	}
	else
	{
		accuracy = 1.0;
	}

	if(c_mat.true_positive_ + c_mat.false_negative_ > 0)
	{
		recall = static_cast<double>(c_mat.true_positive_) / static_cast<double>(c_mat.true_positive_ + c_mat.false_negative_);
	}
	else
	{
		recall = 1.0;
	}

	double f1_score = 2 * precision * recall / (precision + recall);
	out << "precision:\t" << precision << '\n';
	out << "accuracy:\t" << accuracy << '\n';
	out << "recall:\t\t" << recall  << '\n';
	out << "F-score:\t" << f1_score << '\n';
	return out;
}

#endif