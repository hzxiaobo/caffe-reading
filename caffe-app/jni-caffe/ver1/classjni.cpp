#include <jni.h>
#include <string>
#include "classjni.h"
#include "caffeclassifier.h"

map<int, Classifier*> modelPool;
/*
 * Class:     com_netease_is_mi_illegal_image_ni_TargetClassifyJNI
 * Method:    initModel
 * Signature: (Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;IIIFFF)V
 */
JNIEXPORT void JNICALL Java_com_netease_is_mi_illegal_image_ni_TargetClassifyJNI_initModel
(
    JNIEnv *jenv, 
    jclass jcls, 
    jstring jnetPath, 
    jstring jweightsPath, 
    jstring jlabelPath, 
    jint deviceId, 
    jint threadNum,
    jint squareSideLength,  //这个参数是不需要的，可以移除掉，因为crop的size已经是在
    jfloat blueOfMean, 
    jfloat greenOfMean, 
    jfloat redOfMean
) 
{
    try {
        const char* netPathPtr = jenv->GetStringUTFChars(jnetPath, NULL);
        std::string netPath(netPathPtr);
        const char* weightsPathPtr = jenv->GetStringUTFChars(jweightsPath, NULL);
        std::string weightsPath(weightsPathPtr);
        const char* labelPathPtr = jenv->GetStringUTFChars(jlabelPath, NULL);
        std::string labelPath(labelPathPtr);
        for (int i = 0; i < threadNum; ++i ) {
            Classifier* classifier = new Classifier(deviceId, 
                                                    netPath, 
                                                    weightsPath,
                                                    labelPath,
                                                    blueOfMean, 
                                                    greenOfMean, 
                                                    redOfMean);
            //classifier->setResizeSize(squareSideLength);
			//classifier->setClassId(i);
            modelPool.insert(pair<int, Classifier*>(i, classifier));
        }

        std::cout << "net path is: " << netPath << std::endl;


        // realease resources;
        if (netPathPtr) {
            jenv->ReleaseStringUTFChars(jnetPath, netPathPtr);
        }
        if (weightsPathPtr) {
            jenv->ReleaseStringUTFChars(jweightsPath, weightsPathPtr);
        }
        if (labelPathPtr) {
            jenv->ReleaseStringUTFChars(jlabelPath, labelPathPtr);
        }
        
    } catch (Exception e) {
        
    }
}

/*
 * Class:     com_netease_is_mi_illegal_image_ni_TargetClassifyJNI
 * Method:    extractOutputOfLayer
 * Signature: ([BLjava/lang/String;I)[F
 */
JNIEXPORT jfloatArray JNICALL Java_com_netease_is_mi_illegal_image_ni_TargetClassifyJNI_extractOutputOfLayer
(
    JNIEnv *jenv, 
    jclass jcls, 
    jbyteArray jimageBytes, 
    jstring jlayerName, 
    jint threadId
)
{
    jfloatArray joutputArray = NULL;
    try {        
        if (modelPool.count(threadId)) {
            const char* layerNamePtr = jenv->GetStringUTFChars(jlayerName, NULL);        
            Classifier* classifier = modelPool[threadId];
            jsize len = jenv->GetArrayLength(jimageBytes);
            jbyte* bytePtr = jenv->GetByteArrayElements(jimageBytes, 0);
			vector<char> imgVec(len); //have been clear
			// cout << "the length of image is: " << imgVec.size() << " classifier is: " << threadId << endl;
			char* mpDst = &imgVec[0]; //已指向null
			memcpy(mpDst, bytePtr, len);
			Mat img = imdecode(imgVec, -1);
            
            shared_ptr<Blob<float> > outputBlob = classifier->GetLayerOutput(img, string(layerNamePtr));
            
            joutputArray = jenv->NewFloatArray(outputBlob->count());            
            if (joutputArray != NULL) {
                jenv->SetFloatArrayRegion(joutputArray, 0, outputBlob->count(), outputBlob->cpu_data());
            }      

            // realease resources
            imgVec.clear();
            mpDst = NULL;
            if (layerNamePtr != NULL) {
                jenv->ReleaseStringUTFChars(jlayerName, layerNamePtr);
            }
            if (bytePtr != NULL) {
                jenv->ReleaseByteArrayElements(jimageBytes, bytePtr, 0);
            }    
                            
        }
    } catch (Exception e) {

    }
    return joutputArray;
}

/*
 * Class:     com_netease_is_mi_illegal_image_ni_TargetClassifyJNI
 * Method:    forwardNet
 * Signature: ([BI)V
 */
JNIEXPORT void JNICALL Java_com_netease_is_mi_illegal_image_ni_TargetClassifyJNI_forwardNet
(
    JNIEnv *jenv,
    jclass jcls,
    jbyteArray jimageBytes,
    jint threadId)
{
    try {
        if (modelPool.count(threadId)) {
            Classifier* classifier = modelPool[threadId];
            jsize len = jenv->GetArrayLength(jimageBytes);
            jbyte* bytePtr = jenv->GetByteArrayElements(jimageBytes, 0);
            vector<char> imgVec(len); //have been clear
            // cout << "the length of image is: " << imgVec.size() << " classifier is: " << threadId << endl;
            char* mpDst = &imgVec[0]; //已指向null
            memcpy(mpDst, bytePtr, len);
            Mat img = imdecode(imgVec, -1);

            classifier->Forward(img);

            // realease resources
            imgVec.clear();
            mpDst = NULL;

            if (bytePtr != NULL) {
                jenv->ReleaseByteArrayElements(jimageBytes, bytePtr, 0);
            }

        }
    } catch (Exception e) {

    }
}

/*
 * Class:     com_netease_is_mi_illegal_image_ni_TargetClassifyJNI
 * Method:    getBlobOutputByName
 * Signature: (Ljava/lang/String;I)[F
 */
JNIEXPORT jfloatArray JNICALL Java_com_netease_is_mi_illegal_image_ni_TargetClassifyJNI_getBlobOutputByName
(
    JNIEnv *jenv,
    jclass jcls,
    jstring jblobName,
    jint threadId)
{
    jfloatArray joutputArray = NULL;
    try {
        if (modelPool.count(threadId)) {
            const char* blobNamePtr = jenv->GetStringUTFChars(jblobName, NULL);
            Classifier* classifier = modelPool[threadId];

            shared_ptr<Blob<float> > outputBlob = classifier->GetBlobByName(string(blobNamePtr));

            joutputArray = jenv->NewFloatArray(outputBlob->count());
            if (joutputArray != NULL) {
                jenv->SetFloatArrayRegion(joutputArray, 0, outputBlob->count(), outputBlob->cpu_data());
            }

            // realease resources
            if (blobNamePtr != NULL) {
                jenv->ReleaseStringUTFChars(jblobName, blobNamePtr);
            }
        }
    } catch (Exception e) {

    }
    return joutputArray;
}


/*
 * Class:     com_netease_is_mi_illegal_image_ni_TargetClassifyJNI
 * Method:    targetCheck
 * Signature: ([BI)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_com_netease_is_mi_illegal_image_ni_TargetClassifyJNI_targetCheck
(
    JNIEnv *jenv, 
    jclass jcls, 
    jbyteArray jimageBytes, 
    jint threadId)
{
    jstring joutput = 0;
    const char* outputStrPtr = NULL;
    try {
        if (modelPool.count(threadId)) {
            Classifier* classifier = modelPool[threadId];
            jsize len = jenv->GetArrayLength(jimageBytes);
            jbyte* bytePtr = jenv->GetByteArrayElements(jimageBytes, 0);

			vector<char> imgVec(len); //have been clear	
			char* mpDst = &imgVec[0]; //已指向null
			memcpy(mpDst, bytePtr, len);
			Mat img = imdecode(imgVec, -1);

            std::string outputString = classifier->CheckTarget(img);

            outputStrPtr = outputString.c_str();
            if (outputStrPtr != NULL) {
                joutput = jenv->NewStringUTF(outputStrPtr);
            }
            // release resources
            if (bytePtr != NULL) {
                jenv->ReleaseByteArrayElements(jimageBytes, bytePtr, 0);
            }  
            imgVec.clear();
            mpDst = NULL;  
        }
    } catch (Exception e) {

    }
    return joutput;
}

