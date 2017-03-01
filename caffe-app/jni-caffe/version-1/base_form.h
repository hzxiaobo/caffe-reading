#pragma once
#include <iostream>
#include <string>
#include <sstream>
#include <string.h>
#include <stdlib.h>
#include <vector>

template <class Type>
Type baseform_string2num(const std::string& s,Type ipa) {
	std::istringstream is(s);
	Type num;
	is >> num;
	return num;
}

template <class Type>
 Type baseform_min(Type a, Type b){
	if (a > b){
		return b;
	}else{
		return a;
	}
}

template <class Type>
Type baseform_max(Type a, Type b){
	if (a < b){
		return b;
	}else{
		return a;
	}
}




inline int baseform_charSize(char* ipStr, char ipChar) {
	int count = 0;
	for(unsigned int i=0; i< strlen(ipStr) && ipStr[i]!='\0'; i++)
		if(ipStr[i] == ipChar) count++;

	return count;
};

inline string baseform_delExtType(string ipStr) {
    return ipStr.substr(0,ipStr.rfind("."));
};

inline string baseform_int2Str(int ipInt) {
    std::stringstream ss;
    ss<<ipInt;
    return ss.str();
};

inline int baseform_str2Int(string ipStr) {
    int opResult;
    std::stringstream ss;
    ss << ipStr;
    ss >> opResult; //string -> int
    return opResult;
};


struct similarInfo
{
	int id;
	float score;
};