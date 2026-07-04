create table if not exists blogs (
  id serial not null primary key,
  name text not null,
  viewCount int not null
);

create table if not exists authors (
  id serial primary key not null,
  "name" text
);

create table if not exists authors_blogs (
  author_id int not null references authors(id) on delete cascade,
  blog_id int not null references blogs(id) on delete cascade,
  unique(author_id, blog_id)
);

create table if not exists posts (
  id serial not null primary key,
  blog_id int not null references blogs(id) on delete cascade,
  title text not null,
  tags text[] not null default '{}'
);