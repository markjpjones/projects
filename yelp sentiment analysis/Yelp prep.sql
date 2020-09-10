-- create review dataset for filtering
create table yelp_reviews_filtered as 
table yelp_reviews;
commit;

-- remove reviews with a star rating of 3
delete
from yelp_reviews_filtered
where stars = 3;
commit;


-- remove reviews before 2016
delete
from yelp_reviews_filtered
where extract(year from date)::int < 2016;
commit;


------------------------------

-- see reviews by stars
select stars, count(*)
from yelp_reviews_filtered
group by 1
order by 1;

-- see results by year
select extract(year from date), count(*)
from yelp_reviews_filtered
group by 1
order by 1;


-- check the different types of businesses
select categories, count(*)
from yelp_business
where 1=1
--and categories ilike '%restaurant%'
group by 1
having count(*) > 1
order by 2 desc;



-- random select samples based on star rating
select *
from yelp_reviews_filtered TABLESAMPLE SYSTEM(0.5)
where stars = 1
limit 500;


