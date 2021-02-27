# objectDetectionWithYOLO

Bu projede OpenCv'nin çizgi tespit etme algoritmasıyla beraber DarkNet'in önceden eğitilmiş YOLO modelini beraber kullandık. Bu sayede etraftaki nesneleri tanıyabilen ve yoldaki şeritleri tespit edebilen bir uygulama prototipi oluşturmuş olduk. Yapacağınız projelerde fikir edinmeniz veya bu projeyi geliştirmeniz için kodları paylaşıyorum. Kodların üzerinde çalışmadan önce bilmeniz gereken bir kaç şey var;
1. Projede içerisine hem kameradan görüntü almak için hem de masaüstü ekranınızdan görüntü almak için kodlar bulunuyor, masaüstü görüntüsü kodları projeyi geliştrirken farklı videolar üzerinde denemeler için, kamera görüntüsü kodları ise gerçek dünya üzerinde denemeler yapmak isteyenler için yazıldı burada tercih kararı sizin.
2. Fotoğraflardan görüleceği üzere bazı görüntüler 20 FPS civarı iken bazı görüntüler 4 FPS civarı bunun sebebi projede kullandığımız sinir ağı modelinin ağırlıkları, YOLOv3-tiny ağırlıklarını kullandığınız taktirde yüksek fps elde ediyorsunuz ancak modelin tespit edebilme hassasiyeti azalıyor. YOLOv3-spp ağırlıklarını kullandığınızda ise modelin tespit edebilme kabiliyeti artıyor ancak FPS fazlasıyla düşüyor burda yine tercih kararı size kalıyor.

![od1](https://user-images.githubusercontent.com/79279694/109396292-da770380-7941-11eb-896c-bf71b4ce68a8.png)
![od2](https://user-images.githubusercontent.com/79279694/109396311-ed89d380-7941-11eb-8a3b-ab0340067675.png)
![od3](https://user-images.githubusercontent.com/79279694/109396313-ee226a00-7941-11eb-8b3c-45e484612d03.png)




