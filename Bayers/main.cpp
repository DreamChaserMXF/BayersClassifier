#include <iostream>
#include <fstream>
#include <ctime>
#include "BayesClassifier.h"
using std::cout;
using std::ofstream;

int main()
{
	//cout.rdbuf(ofstream("output.txt").rdbuf());
	freopen("output.txt", "w", stdout);
	const char *feature_file = "sina_dataset_4101.txt";
	string feature_name[] = {"��ע","��˿","����","��ע��˿��","��ע���۱�","�û����Ƹ��Ӷ�","΢������","�¾�΢��","΢������ʱ����",
		"ת������","���ӱ���","ƽ��������", "ԭ��΢��������","΢��ƽ������","�������ƶ�","������Ϣ��","��������ƶ�"};
	//bool feature_validity[] = {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	bool feature_validity[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
	//bool feature_validity[] = {1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0};
	size_t feature_num = sizeof(feature_validity) / sizeof(feature_validity[0]);
	//BayesClassifier bayes(feature_num, feature_name, feature_validity);
	BayesClassifier bayes;

	bayes.SetFeatureNum(feature_num);
	bayes.LoadFeatureName(feature_name);
	bayes.LoadFeatureValidity(feature_validity);
	
	cout << "Loading...\n";
	bayes.LoadSamples(feature_file, true);
	cout << "Samples loaded!\n";

	cout << "Regularizing...\n";
	bayes.RegularizeSamples(BayesClassifier::Equalization);
	//bayes.RegularizeSamples(BayesClassifier::Standardization);
	cout << "Regularized!\n";

	cout << "Training...\n";
	bayes.SetTrainMethod(BayesClassifier::BruteForceSearch);
	//bayes.SetTrainMethod(BayesClassifier::KDTreeSearch);
	bayes.Train();
	cout << "Training finished!\n";

	//bayes.SingleTest();

	time_t t0 = time(NULL);
	bayes.CrossValidation(10);
	time_t t1 = time(NULL);
	cout << "time ellapse: " << t1 - t0 << "s\n";

	return 0;
}