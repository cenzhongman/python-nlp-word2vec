docker build -t baike:1.0 .

docker run  \
--name baike \
-v /home/cen/project:/root/project \
-it \
baike:1.0