
use test
go
select userID,[master].dbo.geoplace.name,dbo.rating_final.rating
from dbo.rating_final,[master].dbo.geoplace 
where dbo.rating_final.placeID =[master].dbo.geoplace.placeID and userID='U1011'