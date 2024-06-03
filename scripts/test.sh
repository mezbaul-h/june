#!/usr/bin/env bash

TESTING=true coverage run --source='.' manage.py test
coverage report
coverage xml
