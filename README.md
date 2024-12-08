# CS-GY-6613-Project

1. docker pull mongo

2. docker run -d --name mongodb -p 27017:27017 mongo

3. docker ps

4. docker exec -it mongodb bash

5. mongosh

6. show dbs

7. use github_scraper

8. run scraper_github.py
   - python crawler/scraper_github.py
     
9. show collections

10. db.repositories.find().pretty()

11. use medium_scraper

12. run scraper_medium.py
    - pytho crawler/scraper_medium.py

13. db.repositories.find({}, { repository_id: 1 }).pretty()

14. copy and put it in cleaning.py

15. use github_scraper

16. db.repositories.find({}, { repository_id: 1 }).pretty()

17. copy and put it in cleaning.py

18. python3 cleaning.py

19. python3 inference_pipeline.py

20. docker pull qdrant/qdrant

21. docker run -d --name qdrant -p 6333:6333 -v qdrant-storage:/qdrant/storage qdrant/qdrant

22. python3 push_qna_to_qdrant.py

23. python3 serve_model.py

24. in new terminal
    - python3 rag_api.py

25. in new terminal
    - python3 rag_gradio_app.py

26. http://0.0.0.0:7860
