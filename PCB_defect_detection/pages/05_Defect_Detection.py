import streamlit as st
from streamlit_function import func

func.set_title(__file__)

with st.sidebar:
    radio_sidebar = st.radio(label="", label_visibility='collapsed', options=["흑백 데이터셋", "실물 데이터셋"])
    
if radio_sidebar == '흑백 데이터셋':
    tab1 = st.tabs(['데이터 전처리', '학습시킨 결함 종류', '데이터 증강', '모델 학습', '모델 평가', '모델 테스트'])
    with tab1[0]:
        st.subheader('Data Preprocessing')

        func.image_resize('data_preprocessing_01.png', __file__, 500)
    with tab1[1]:
        st.subheader('Defect Type')

        radio_defect = st.radio(label="", label_visibility='collapsed', options=["Open(회로 개방)", "Short(회로 단락)", "Copper(구리 결함)", "Mousebit", "Pin-hole", "Spur"], horizontal=True)
        if radio_defect == "Open(회로 개방)":
            func.image_resize('defect_open.png', __file__, 400)
        elif radio_defect == "Short(회로 단락)":
            func.image_resize('defect_short.png', __file__, 400)
        elif radio_defect == "Copper(구리 결함)":
            func.image_resize('defect_copper.png', __file__, 400)
        elif radio_defect == "Mousebit":
            func.image_resize('defect_mousebit.png', __file__, 400)
        elif radio_defect == "Pin-hole":
            func.image_resize('defect_pinhole.png', __file__, 400)
        elif radio_defect == "Spur":
            func.image_resize('defect_spur.png', __file__, 400)
    with tab1[2]:
        st.subheader('Data Augmentation')

        col1, col2, col3, _ = st.columns(4)

        with col1:
            st.text("상하좌우 대칭")
            func.image_resize('data_augmentation_01.png', __file__, 350)
        with col2:
            st.text("90° 회전")
            st.markdown('\n')
            st.markdown('\n')
            func.image_resize('data_augmentation_02.png', __file__, 150)
        with col3:            
            st.text("크기 조절")
            func.image_resize('data_augmentation_03.png', __file__, 250)

    with tab1[3]:
        st.subheader('Model Training')

        radio_train = st.radio(label="", label_visibility='collapsed', options=["전이 학습", "모델 설계", "모델 학습"], horizontal=True)
        if radio_train == '전이 학습':
            st.page_link('./pages/98_Source.py', label='Source', icon="🚨")
            col1, col2 = st.columns(2)
            with col1:
                func.image_resize('vgg16_model.png', __file__, 400)
            with col2:
                st.markdown("""### **기대효과**
                                \n- 학습 효율성 향상
                                \n- 모델 성능 향상
                                \n- 모델 일반화
                                \n- 자원 절약""")
                st.markdown("\n")
                st.markdown("실제로 효과가 있는지 확인해 보기 위해 모델을 여러 가지 만들어서 비교해 보았습니다.")
        elif radio_train == '모델 설계':
            with st.expander('Source Code', expanded=True):
                st.code("""
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
                
datagenAug = ImageDataGenerator(
    rotation_range=90,
	zoom_range=0.15,
	horizontal_flip=True,
	vertical_flip=True,
	validation_split=0.2
)

path = "./PCBData/defects"
trainGen = datagenAug.flow_from_directory(
    path, 
    classes=["open", "short", "mousebit", 
             "spur", "copper", "pin-hole"],
    target_size=(224, 224), 
    color_mode='grayscale',
    class_mode="sparse",
    batch_size=64, 
    subset="training")

testGen = datagenAug.flow_from_directory(
    path, 
    classes=["open", "short", "mousebit", 
             "spur", "copper", "pin-hole"],
    target_size=(224, 224), 
    color_mode='grayscale',
    class_mode="sparse",
    batch_size=64, 
    subset="validation")
                
model = Sequential()
""")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("### 임의로 층 설계")
                with st.expander('Source Code', expanded=True):
                    st.code(""" 
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', input_shape=(224,224,1)))
model.add(MaxPooling2D(2))
model.add(Conv2D(64, kernel_size=(3,3),activation='relu', padding='same'))
model.add(MaxPooling2D(2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(6, activation='softmax'))
""")
                with st.expander('Model Summary'):
                    st.image("./streamlit_images/defect_detection/not_use_vgg16_model_summary.png", use_column_width=True)
            with col2:
                st.markdown("### VGG16 모델과 비슷하게 층 설계")
                with st.expander('Source Code', expanded=True):
                    st.code("""
model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', input_shape=(224,224,1)))
model.add(Conv2D(64, kernel_size=(3,3),activation='relu', padding='same'))
model.add(MaxPooling2D(2))
model.add(Conv2D(128, kernel_size=(3,3),activation='relu', padding='same'))
model.add(Conv2D(128, kernel_size=(3,3),activation='relu', padding='same'))
model.add(MaxPooling2D(2))
model.add(Conv2D(256, kernel_size=(3,3),activation='relu', padding='same'))
model.add(Conv2D(256, kernel_size=(3,3),activation='relu', padding='same'))
model.add(Conv2D(256, kernel_size=(3,3),activation='relu', padding='same'))
model.add(MaxPooling2D(2))
model.add(Conv2D(512, kernel_size=(3,3),activation='relu', padding='same'))
model.add(Conv2D(512, kernel_size=(3,3),activation='relu', padding='same'))
model.add(Conv2D(512, kernel_size=(3,3),activation='relu', padding='same'))
model.add(MaxPooling2D(2))
model.add(Conv2D(512, kernel_size=(3,3),activation='relu', padding='same'))
model.add(Conv2D(512, kernel_size=(3,3),activation='relu', padding='same'))
model.add(Conv2D(512, kernel_size=(3,3),activation='relu', padding='same'))
model.add(MaxPooling2D(2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(6, activation='softmax'))
""")
                with st.expander('Model Summary'):
                    st.image("./streamlit_images/defect_detection/not_use_vgg16_2_model_summary.png", use_column_width=True)
            with col3:
                st.markdown("### VGG16 모델을 넣어서 설계")
                with st.expander('Source Code', expanded=True):
                    st.code("""
vgg_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224,224,3)))
                        
for layer in vgg_model.layers[:-1]:
    layer.trainable = False
                        
x = vgg_model.output
x = Flatten()(x) 
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(6, activation='softmax')(x)  
model = Model(inputs=vgg_model.input, outputs=predictions)
""")
                with st.expander('Model Summary'):
                    st.image("./streamlit_images/defect_detection/use_vgg16_model_summary.png", use_column_width=True)
            
        elif radio_train == '모델 학습':
            st.code("""
model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
                    
checkpoint = ModelCheckpoint('./model/model.keras',
                             monitor='val_accuracy', 
                             verbose=1, 
                             save_best_only=True, 
                             save_weights_only=False)
                    
early_stopping = EarlyStopping(monitor='val_accuracy',
                      patience=5, 
                      verbose=1)
                    
history = model.fit(trainGen,
                    epochs=50,
                    validation_data=testGen,
                    callbacks=[checkpoint, early_stopping])
""")

    with tab1[4]:
        st.subheader('Model Evaluation')
        radio_evaluation = st.radio(label="", label_visibility='collapsed', options=["학습 결과 시각화", "혼동 행렬", "도출 결과"], horizontal=True)
        if radio_evaluation == '학습 결과 시각화':
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("max_accuracy : 0.9594358205795288")
                st.markdown("max_val_accuracy : **:red[0.9905047416687012]**")
                st.markdown("min_loss : 0.13344751298427582")
                st.markdown("min_cal_loss : 0.03660937026143074")
                st.image('./streamlit_images/defect_detection/accuracy_not_use_vgg16.png', use_column_width=True)
                st.image('./streamlit_images/defect_detection/loss_not_use_vgg16.png', use_column_width=True)
            with col2:
                st.markdown("max_accuracy : 0.9885172247886658")
                st.markdown("max_val_accuracy : **:red[0.998000979423523]**")
                st.markdown("min_loss : 0.04347741976380348")
                st.markdown("min_cal_loss : 0.008867921307682991")
                st.image('./streamlit_images/defect_detection/accuracy_not_use_vgg16_2.png', use_column_width=True)
                st.image('./streamlit_images/defect_detection/loss_not_use_vgg16_2.png', use_column_width=True)
            with col3:
                st.markdown("max_accuracy : 0.9601846933364868")
                st.markdown("max_val_accuracy : **:red[0.9940029978752136]**")
                st.markdown("min_loss : 0.16124936938285828")
                st.markdown("min_cal_loss : 0.03154732286930084")
                st.image('./streamlit_images/defect_detection/accuracy_use_vgg16.png', use_column_width=True)
                st.image('./streamlit_images/defect_detection/loss_use_vgg16.png', use_column_width=True)
        elif radio_evaluation == '혼동 행렬':
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("### accuracy : **:red[0.930]** 5347326336831")
                st.image('./streamlit_images/defect_detection/confusion_matrix_not_use_vgg16.png', use_column_width=True)
            with col2:
                st.markdown("### accuracy : **:red[0.997]** 0014992503748")
                st.image('./streamlit_images/defect_detection/confusion_matrix_not_use_vgg16_2.png', use_column_width=True)
            with col3:
                st.markdown("### accuracy : **:red[0.985]** 0074962518741")
                st.image('./streamlit_images/defect_detection/confusion_matrix_use_vgg16_2.png', use_column_width=True)
        elif radio_evaluation == '도출 결과':
            col1, col2, _ = st.columns(3)
            with col1:
                st.markdown('임의로 층 설계한 모델')
                st.markdown("### **:red[0.930]** 5347326336831")
                st.markdown('VGG16 모델을 넣어서 설계')
                st.markdown("### **:red[0.997]** 0014992503748")
                st.markdown('VGG16을 사용한 모델')
                st.markdown("### **:red[0.985]** 0074962518741")
            with col2:
                st.markdown("첫 번째 모델은 예측률이 부족하고")
                st.markdown("두 번째 모델은 과적합이 심해 일반화가 안 되어있습니다.")
                st.markdown("\n")
                st.markdown("""**VGG16을 사용한 모델의 경우**
                                \n- 첫 번째 모델보다 성능이 향상된 모습
                                \n- 두 번째 모델보다 일반화가 된 모습
                                \n- 다른 두 모델보다 크기가 작은 모습""")
                func.image_resize('model_size.png', __file__, 100)

    with tab1[5]:
        st.subheader('Model Test')

    col1 , col2, col3 = st.columns(3)    
elif radio_sidebar == '실물 데이터셋':
    st.text("고양이")