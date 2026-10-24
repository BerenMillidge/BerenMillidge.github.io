---
layout: page
title: Archive
subtitle: All writing, organized by year
permalink: /archive/
---

{% assign archive_posts = site.posts | sort: "date" | reverse %}
{% assign archive_years = archive_posts | group_by_exp: "post", "post.date | date: '%Y'" %}

<style>
  /* Compact archive only: does not change the rest of the blog. */
  .archive-page {
    max-width: 780px;
    margin: 0 auto;
    font-family: 'Open Sans', 'Helvetica Neue', Helvetica, Arial, sans-serif;
  }

  .archive-page .archive-summary {
    margin: 0 0 12px;
    color: #666;
    font-size: 13px;
    line-height: 1.5;
  }

  .archive-page .archive-years {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 8px 15px;
    margin: 0 0 30px;
    font-size: 13px;
    line-height: 1.5;
  }

  .archive-page .archive-years-label {
    color: #666;
  }

  .archive-page .archive-years a {
    text-decoration: none;
  }

  .archive-page .archive-years a:hover,
  .archive-page .archive-years a:focus {
    text-decoration: underline;
  }

  .archive-page .archive-year {
    margin: 0 0 28px;
    scroll-margin-top: 110px;
  }

  .archive-page .archive-year h2 {
    margin: 0 0 8px;
    padding-bottom: 7px;
    border-bottom: 1px solid #e9e9e9;
    font-size: 22px;
    font-weight: 700;
    line-height: 1.3;
  }

  .archive-page .archive-count {
    margin-left: 6px;
    color: #666;
    font-size: 12px;
    font-weight: 400;
  }

  .archive-page .archive-posts {
    padding: 0;
    margin: 0;
    list-style: none;
  }

  .archive-page .archive-item {
    display: flex;
    align-items: baseline;
    gap: 14px;
    padding: 4px 0;
    margin: 0;
    font-size: 14px; /* Site's usual body copy is 18px. */
    line-height: 1.45;
  }

  .archive-page .archive-date {
    flex: 0 0 40px;
    color: #666;
    font-size: 12px;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
  }

  .archive-page .archive-item a {
    color: #404040;
    text-decoration: none;
  }

  .archive-page .archive-item a:hover,
  .archive-page .archive-item a:focus {
    color: {{ site.hover-col | default: '#0085A1' }};
    text-decoration: underline;
  }

  @media (max-width: 500px) {
    .archive-page .archive-item {
      gap: 9px;
      font-size: 13px;
    }

    .archive-page .archive-year h2 {
      font-size: 20px;
    }
  }
</style>

<div class="archive-page">
  <p class="archive-summary">{{ archive_posts.size }} posts, newest first.</p>

  <nav class="archive-years" aria-label="Jump to a year">
    <span class="archive-years-label">Jump to:</span>
    {% for year in archive_years %}
      <a href="#archive-year-{{ year.name }}">{{ year.name }}</a>
    {% endfor %}
  </nav>

  {% for year in archive_years %}
  <section class="archive-year" id="archive-year-{{ year.name }}" aria-labelledby="archive-heading-{{ year.name }}">
    <h2 id="archive-heading-{{ year.name }}">
      {{ year.name }}
      <span class="archive-count">{{ year.items.size }} posts</span>
    </h2>

    <ul class="archive-posts">
      {% for post in year.items %}
      <li class="archive-item">
        <time class="archive-date" datetime="{{ post.date | date: '%Y-%m-%d' }}">{{ post.date | date: '%m/%d' }}</time>
        <a href="{{ post.url | relative_url }}">{{ post.title | escape }}</a>
      </li>
      {% endfor %}
    </ul>
  </section>
  {% endfor %}
</div>
