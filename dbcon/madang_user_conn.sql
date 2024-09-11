select * from book;
select bookname, price from book;
select price, bookname from book;
select bookid, bookname, publisher, price from book;
select * from book;
select publisher from book;
select distinct publisher from book;
select * from book;
select * from book where price < 20000;
select * from book where price Between 10000 and 20000;
select * from book where price >= 10000 and price <= 20000;
select * from book where publisher in ('굿스포츠', '대한미디어');
select * from book where publisher not in ('굿스포츠', '대한미디어');
select bookname, publisher from book where bookname like '축구의 역사';
select bookname, publisher from book where bookname like '%축구%';
select * from book where bookname like '_구%';
select * from book where bookname like '%축구%' and price >= 20000;
select * from book where publisher='굿스포츠' or publisher='대한미디어';
select * from book order by bookname;
select * from book order by price, bookname;
select * from book order by price desc, publisher asc;
select sum(saleprice) from orders;
select sum(saleprice) as 총매출 from orders;
select sum(saleprice) as 총매출 from orders where custid=2;
select sum(saleprice) as Total,
       avg(saleprice) as Average,
       min(saleprice) as Minimum,
       max(saleprice) as Maximum
from orders;
select count(*) from orders;

select custid, count(*) as 도서수량, sum(saleprice) as 총액
from orders
group by custid;

select custid, count(*) as 도서수량
from orders 
where saleprice >= 8000
group by custid
having count(*) >= 2;

select * from book;

/* 도서번호가 1인 도서의 이름 */
select bookid, bookname from book where bookid=1;

/* 가격이 20000원 이상인 도서의 이름 */
select price, bookname from book where price >= 20000

/* 박지성의 총 구매액(박지성의 고객번호는 1번으로 놓고 작성 */
select * from orders;
select * from customer;

/* 6000 + 21000 + 12000 = 39000 */
select sum(saleprice) as 총구매액 from orders where custid=1;


/* 박지성이 구매한 도서의 수(박지성의 고객번호는 1번으로 놓고 작성) */
select count(custid) as 구매도서수 from orders where custid=1;

/* 마당서점 도서의 총 개수 */
select count(bookid) as 도서총개수 from book;

/* 마당서점에 도서를 출고하는 출판사의 총 개수 */
select sum(distinct count(publisher)) from book
group by publisher;

/* 모든 고객의 이름, 주소 */
select * from customer;
select distinct name, address from customer;

/* 2014년 7월 4일~7월 7일 사이에 주문 받은 도서의 주문번호 */
select * from orders;
select orderid, orderdate from orders 
where orderdate >= to_date(200704,'YYMMDD') and orderdate <= to_date(200707,'YYMMDD')

/* 2014년 7월 4일~7월 7일 사이에 주문 받은 도서를 제외한 도서의 주문번호 */
select bookid, orderdate from orders 
where orderdate < to_date(200704,'YYMMDD') or orderdate > to_date(200707,'YYMMDD')

/* 성이 '김'씨인 고객의 이름과 주소 */
select name, address from customer
where name like '김%';

/* 성이 '김'씨이고 이름이 '아'로 끝나는 고객의 이름과 주소 */
select name, address from customer
where name like '김%' and name like '%아';

