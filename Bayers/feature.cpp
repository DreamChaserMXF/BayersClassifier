#include "Feature.h"
#include <istream>
#include <ostream>
using std::istream;
using std::ostream;


Feature::Feature(int attr_count, bool isLabeled) : values_(attr_count), isLabeled_(isLabeled), real_type_(VOID), predicted_type_(VOID)
{
}

bool Feature::operator < (const Feature &right)
{
	return id_ < right.id_;
}
ostream& operator << (ostream &out, const Feature &feature)
{
	out << feature.id_;
	for(vector<double>::const_iterator iter = feature.values_.begin(); iter != feature.values_.end(); ++iter)
	{
		out << '\t' << *iter;
	}
	if(true == feature.isLabeled_)
	{
		out << '\t' << feature.real_type_;
	}
	else
	{
		out << '\t' << feature.predicted_type_;
	}
	out << '\n';
	return out;
}

istream& operator >> (istream &in, Feature &feature)
{
	in >> feature.id_;
	if("quit" == feature.id_)
	{
		in.setstate(std::ios::badbit);
		return in;
	}
	for(vector<double>::size_type i = 0; i < feature.values_.size(); ++i)
	{
		in >> feature.values_.at(i);
	}
	if(true == feature.isLabeled_)
	{
		//int *real_type_p = reinterpret_cast<int*>(&feature.real_type_);
		//in >> *real_type_p;
		in >> *(reinterpret_cast<int*>(&feature.real_type_));
		if (Feature::TRUE != feature.real_type_
		 && Feature::FALSE != feature.real_type_)
		{
			feature.real_type_ = Feature::VOID;
		}
	}
	return in;
}