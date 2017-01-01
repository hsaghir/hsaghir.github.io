---
layout: archive
permalink: /
title: ""
---

<div class="tiles">
<h3>Deep Learning</h3>
{% for post in site.categories.data_science %}
  {% include post-grid.html %}
{% endfor %}

<h3>Job Hunting</h3>
{% for post in site.categories.job %}
  {% include post-grid.html %}
{% endfor %}

</div>

<div class="tiles">
<h3>Philosophy</h3>
{% for post in site.categories.philosophy %}
  {% include post-grid.html %}
{% endfor %}
</div>