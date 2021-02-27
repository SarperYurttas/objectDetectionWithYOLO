# objectDetectionWithYOLO

English description:
In this project we used OpenCv's line detection algorithm and DarkNet's pre-trained YOLO model. Through we have an object recognition and line detection application prototype. I'm sharing this project so you can get inspiration for your project or you can improve this project. You need to know something before work on this codes;
1. There is either screen capture or camera use codes in this project. Screen capture for testing different videos in your desktop easily. Webcam images for implementation real world problems it is your choice.
2. As you can see some videos has nearby 20 FPS but other some videos has just 4 FPS the reason for this the neural network model has different weights and we use two of them. If you use YOLOv3-tiny weight you will see 20 FPS but this weights has restricted recognition ability. If you use YOLOv3-spp weights you will see 4 FPS and more recognition ability, choose what you need.
3. You can download weights, cfg and names files in DarkNet's official website.


Türkçe açıklama:
Bu projede OpenCv'nin çizgi tespit etme algoritmasıyla beraber DarkNet'in önceden eğitilmiş YOLO modelini beraber kullandık. Bu sayede etraftaki nesneleri tanıyabilen ve yoldaki şeritleri tespit edebilen bir uygulama prototipi oluşturmuş olduk. Yapacağınız projelerde fikir edinmeniz veya bu projeyi geliştirmeniz için kodları paylaşıyorum. Kodların üzerinde çalışmadan önce bilmeniz gereken bir kaç şey var;
1. Projede içerisine hem kameradan görüntü almak için hem de masaüstü ekranınızdan görüntü almak için kodlar bulunuyor, masaüstü görüntüsü kodları projeyi geliştrirken farklı videolar üzerinde denemeler için, kamera görüntüsü kodları ise gerçek dünya üzerinde denemeler yapmak isteyenler için yazıldı. Bu konu hakkında tercih kararı sizin.
2. Fotoğraflardan görüleceği gibi bazı görüntüler 20 FPS civarı iken bazı görüntüler 4 FPS civarı bunun sebebi projede kullandığımız sinir ağı modelinin ağırlıklarının farkıdır. Öyle ki YOLOv3-tiny ağırlıklarını kullandığınız taktirde yüksek fps elde ediyorsunuz ancak modelin tespit edebilme hassasiyeti azalıyor. YOLOv3-spp ağırlıklarını kullandığınızda ise modelin tespit edebilme kabiliyeti artıyorsa bile FPS fazlasıyla düşüyor burda yine tercih kararı size kalıyor.
3. Weights, cfg ve names dosyalarını DarkNet'in resmi sitesinden edinebilirsiniz.

![od1](https://user-images.githubusercontent.com/79279694/109396292-da770380-7941-11eb-896c-bf71b4ce68a8.png)
![od2](https://user-images.githubusercontent.com/79279694/109396311-ed89d380-7941-11eb-8a3b-ab0340067675.png)
![od3](https://user-images.githubusercontent.com/79279694/109396313-ee226a00-7941-11eb-8b3c-45e484612d03.png)




